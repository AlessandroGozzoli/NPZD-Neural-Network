# =============================================================================
# data_generator.py
# Monte Carlo NPZD simulation → supervised training dataset.
#
# Key improvements over original:
#   1. N-conserving initial conditions: sample total_N, partition among
#      compartments, so N+P+Z+D = const from t=0.
#   2. Targets are STATE INCREMENTS Δs = s_{t+1} − s_t (delta formulation).
#   3. Stability filters are minimal and physically meaningful: the Radau
#      solver + analytical Jacobian should produce clean trajectories.
#   4. Rejection reasons are counted and reported.
# =============================================================================

import os
import copy
import numpy as np
from tqdm import tqdm

from config import ODE_PARAMS, DATA_GEN, SOLVER
from npzd_ode import run_npzd


# =============================================================================
# Initial condition sampling (N-conserving)
# =============================================================================

def sample_initial_condition(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """
    Sample winter initial conditions [N0, P0, Z0, D0] independently.
    High N, low biology — the natural state at the start of the annual cycle.
    The system is open so there is no N budget constraint.
    """
    N0 = rng.uniform(*cfg["N0_range"])
    P0 = rng.uniform(*cfg["P0_range"])
    Z0 = rng.uniform(*cfg["Z0_range"])
    D0 = rng.uniform(*cfg["D0_range"])
    return np.array([N0, P0, Z0, D0])


# =============================================================================
# Parameter perturbation
# =============================================================================

def perturb_params(base: dict, rng: np.random.Generator, cfg: dict) -> dict:
    """Uniformly perturb selected ecological parameters within ±frac."""
    params = copy.deepcopy(base)
    frac   = cfg["param_perturb_frac"]
    for key in cfg["perturb_params"]:
        nominal = base[key]
        params[key] = rng.uniform(nominal * (1.0 - frac),
                                   nominal * (1.0 + frac))
    # Safety: alpha + beta must be < 1 after perturbation
    while params["alpha"] + params["beta"] >= 1.0:
        params["alpha"] *= 0.9
        params["beta"]  *= 0.9
    return params


# =============================================================================
# Trajectory quality check
# =============================================================================

def _is_acceptable(result: dict, cfg: dict) -> tuple:
    """
    Return (True, '') or (False, reason_str).
    Only checks that are valid for an OPEN system are kept.
    N conservation drift is NOT checked — the mixing term actively
    adds external nitrogen, so total N drifting is correct physics.
    """
    states = result["states"]

    # 1. Blow-up or NaN
    if np.any(np.isnan(states)) or np.any(states > cfg["max_concentration"]):
        return False, "blow-up or NaN"

    # 2. Negative values beyond floating-point tolerance
    if np.any(states < cfg["min_state_value"]):
        worst = states.min()
        return False, f"negative concentration {worst:.2e}"

    return True, ""


# =============================================================================
# Extract one-step transition pairs (delta formulation)
# =============================================================================

def extract_pairs(result: dict) -> tuple:
    """
    Build input/target arrays from a single trajectory.

    Input  X[i] = [N_t, P_t, Z_t, D_t, I_t, T_t]   (current state + forcing)
    Target y[i] = [ΔN, ΔP, ΔZ, ΔD]                  (state INCREMENT)

    The delta formulation means the NN learns small corrections rather
    than absolute next-state values, which is both easier and physically
    more stable in autoregressive rollout.
    """
    states  = result["states"]    # (T, 4)
    forcing = result["forcing"]   # (T, 2)

    # Inputs:  state and forcing at t
    X = np.concatenate([states[:-1], forcing[:-1]], axis=1)  # (T-1, 6)
    # Targets: increment from t to t+1
    y = states[1:] - states[:-1]                              # (T-1, 4)

    return X, y


# =============================================================================
# Main dataset generation
# =============================================================================

def generate_dataset(
    n_trajectories : int  = None,
    max_samples    : int  = None,
    random_seed    : int  = None,
    data_dir       : str  = None,
    verbose        : bool = True,
) -> tuple:
    """
    Run Monte Carlo NPZD simulations and build the training dataset.

    Returns (X, y) arrays and saves them as .npy files.
    """
    cfg  = DATA_GEN
    n_trajectories = n_trajectories or cfg["n_trajectories"]
    max_samples    = max_samples    or cfg["max_samples"]
    random_seed    = random_seed    or cfg["random_seed"]
    data_dir       = data_dir       or cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    X_list, y_list = [], []
    n_failed  = 0
    reject_log: dict = {}

    itr = tqdm(range(n_trajectories), desc="Generating trajectories") \
          if verbose else range(n_trajectories)

    for _ in itr:
        y0     = sample_initial_condition(rng, cfg)
        params = perturb_params(ODE_PARAMS, rng, cfg)
        result = run_npzd(y0, params=params, solver_cfg=SOLVER)

        if not result["success"]:
            n_failed += 1
            key = "solver failed"
            reject_log[key] = reject_log.get(key, 0) + 1
            continue

        ok, reason = _is_acceptable(result, cfg)
        if not ok:
            n_failed += 1
            key = reason.split(" ")[0]
            reject_log[key] = reject_log.get(key, 0) + 1
            continue

        Xb, yb = extract_pairs(result)
        X_list.append(Xb)
        y_list.append(yb)

    if verbose:
        n_ok = len(X_list)
        print(f"\nSuccessful : {n_ok} / {n_trajectories}")
        print(f"Rejected   : {n_failed}")
        if reject_log:
            print("Breakdown  :")
            for k, v in sorted(reject_log.items(), key=lambda x: -x[1]):
                print(f"  {v:>5}  {k}")

    if not X_list:
        raise RuntimeError("All trajectories were rejected. "
                           "Check ODE parameters and stability filters.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)

    if verbose:
        print(f"\nTotal pairs before subsampling: {len(X):,}")

    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]
        if verbose:
            print(f"Subsampled to: {len(X):,}")

    np.save(os.path.join(data_dir, "X.npy"), X)
    np.save(os.path.join(data_dir, "y.npy"), y)

    if verbose:
        _print_stats(X, y)
        print(f"\nSaved → {data_dir}/X.npy  {X.shape}")
        print(f"Saved → {data_dir}/y.npy  {y.shape}")

    return X, y


# =============================================================================
# Evaluation trajectory generation
# =============================================================================

def generate_trajectories_for_eval(
    n_trajectories : int  = None,
    random_seed    : int  = 999,
    data_dir       : str  = None,
    verbose        : bool = True,
) -> list:
    """
    Generate complete trajectories for autoregressive rollout evaluation.
    Saved as a list of dicts to data/eval_trajectories.npy.
    """
    from config import EVAL
    cfg      = DATA_GEN
    n_trajectories = n_trajectories or EVAL["n_eval_trajectories"]
    data_dir = data_dir or cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    rng          = np.random.default_rng(random_seed)
    trajectories = []

    itr = tqdm(range(n_trajectories * 2), desc="Generating eval trajectories") \
          if verbose else range(n_trajectories * 2)

    for _ in itr:
        if len(trajectories) >= n_trajectories:
            break
        y0     = sample_initial_condition(rng, cfg)
        params = perturb_params(ODE_PARAMS, rng, cfg)
        result = run_npzd(y0, params=params, solver_cfg=SOLVER)

        if not result["success"]:
            continue
        ok, _ = _is_acceptable(result, cfg)
        if not ok:
            continue

        result["y0"]     = result["states"][0].copy()
        result["params"] = params
        trajectories.append(result)

    from config import EVAL as EVAL_CFG
    out_path = EVAL_CFG["eval_traj_file"]
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    np.save(out_path, trajectories, allow_pickle=True)

    if verbose:
        print(f"\nEval trajectories: {len(trajectories)} saved → {out_path}")

    return trajectories


# =============================================================================
# Utility
# =============================================================================

def _print_stats(X: np.ndarray, y: np.ndarray) -> None:
    fn = ["N", "P", "Z", "D", "I", "T"]
    tn = ["ΔN", "ΔP", "ΔZ", "ΔD"]
    print("\n--- Input feature statistics ---")
    for i, n in enumerate(fn):
        print(f"  {n:4s} mean={X[:,i].mean():8.4f}  std={X[:,i].std():7.4f}"
              f"  min={X[:,i].min():8.4f}  max={X[:,i].max():8.4f}")
    print("--- Target (Δ) statistics ---")
    for i, n in enumerate(tn):
        print(f"  {n:4s} mean={y[:,i].mean():8.4f}  std={y[:,i].std():7.4f}"
              f"  min={y[:,i].min():8.4f}  max={y[:,i].max():8.4f}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    X, y = generate_dataset(verbose=True)
    generate_trajectories_for_eval(verbose=True)
