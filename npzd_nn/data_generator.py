# =============================================================================
# data_generator.py
# Generates the supervised training dataset by running many NPZD ODE
# trajectories with randomly sampled initial conditions and perturbed
# ecological parameters, then extracting consecutive one-step pairs.
#
# Output:
#   X.npy  — input array  (n_samples, 6):  [N_t, P_t, Z_t, D_t, I_t, T_t]
#   y.npy  — target array (n_samples, 4):  [N_{t+1}, P_{t+1}, Z_{t+1}, D_{t+1}]
# =============================================================================

import os
import copy
import numpy as np
from tqdm import tqdm

from config import ODE_PARAMS, DATA_GEN, SOLVER
from npzd_ode import run_npzd


# =============================================================================
# Parameter perturbation
# =============================================================================

def perturb_params(base_params: dict, rng: np.random.Generator, cfg: dict) -> dict:
    """
    Return a copy of base_params with selected parameters uniformly perturbed
    within ±param_perturb_frac of their nominal values.
    """
    params = copy.deepcopy(base_params)
    frac = cfg["param_perturb_frac"]

    for key in cfg["perturb_params"]:
        nominal = base_params[key]
        lo = nominal * (1.0 - frac)
        hi = nominal * (1.0 + frac)
        params[key] = rng.uniform(lo, hi)

    return params


# =============================================================================
# Initial condition sampling
# =============================================================================

def sample_initial_condition(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """
    Draw a random initial state [N0, P0, Z0, D0] from the configured ranges.
    """
    N0 = rng.uniform(*cfg["N0_range"])
    P0 = rng.uniform(*cfg["P0_range"])
    Z0 = rng.uniform(*cfg["Z0_range"])
    D0 = rng.uniform(*cfg["D0_range"])
    return np.array([N0, P0, Z0, D0])


# =============================================================================
# Extract one-step transition pairs from a single trajectory
# =============================================================================

def extract_pairs(result: dict) -> tuple:
    """
    From a solved trajectory, extract all consecutive (state_t, forcing_t) →
    state_{t+1} pairs.

    Parameters
    ----------
    result : dict returned by run_npzd()

    Returns
    -------
    X : np.ndarray, shape (n_steps-1, 6)   [N_t, P_t, Z_t, D_t, I_t, T_t]
    y : np.ndarray, shape (n_steps-1, 4)   [N_{t+1}, P_{t+1}, Z_{t+1}, D_{t+1}]
    """
    states  = result["states"]    # (n_steps, 4)
    forcing = result["forcing"]   # (n_steps, 2)

    # Inputs: state at t=0..T-2, forcing at t=0..T-2
    X = np.concatenate([states[:-1], forcing[:-1]], axis=1)   # (n_steps-1, 6)
    # Targets: state at t=1..T-1
    y = states[1:]                                             # (n_steps-1, 4)

    return X, y


# =============================================================================
# Main data generation routine
# =============================================================================

def generate_dataset(
    n_trajectories  : int  = None,
    max_samples     : int  = None,
    random_seed     : int  = None,
    data_dir        : str  = None,
    verbose         : bool = True,
) -> tuple:
    """
    Run Monte Carlo NPZD simulations and build the supervised dataset.

    Returns
    -------
    X : np.ndarray (n_samples, 6)
    y : np.ndarray (n_samples, 4)
    Also saves X and y as .npy files in data_dir.
    """
    cfg = DATA_GEN

    n_trajectories = n_trajectories or cfg["n_trajectories"]
    max_samples    = max_samples    or cfg["max_samples"]
    random_seed    = random_seed    or cfg["random_seed"]
    data_dir       = data_dir       or cfg["data_dir"]

    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    X_list, y_list = [], []
    n_failed = 0

    iterator = tqdm(range(n_trajectories), desc="Generating trajectories") \
               if verbose else range(n_trajectories)

    for i in iterator:
        # Sample initial conditions and perturbed parameters
        y0     = sample_initial_condition(rng, cfg)
        params = perturb_params(ODE_PARAMS, rng, cfg)

        # Solve ODE
        result = run_npzd(y0, params=params, solver_cfg=SOLVER)

        if not result["success"]:
            n_failed += 1
            continue

        # Check for numerical blow-up (concentrations should stay reasonable)
        if np.any(result["states"] > 1e4) or np.any(np.isnan(result["states"])):
            n_failed += 1
            continue

        # Extract one-step pairs
        X_traj, y_traj = extract_pairs(result)
        X_list.append(X_traj)
        y_list.append(y_traj)

    if verbose:
        print(f"\nSuccessful trajectories : {len(X_list)} / {n_trajectories}")
        print(f"Failed / rejected       : {n_failed}")

    # Concatenate all pairs
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)

    if verbose:
        print(f"Total one-step pairs before subsampling: {len(X):,}")

    # Subsample if needed
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]
        if verbose:
            print(f"Subsampled to: {len(X):,} pairs")

    # Save
    X_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    np.save(X_path, X)
    np.save(y_path, y)

    if verbose:
        print(f"\nDataset saved:")
        print(f"  X -> {X_path}  shape={X.shape}")
        print(f"  y -> {y_path}  shape={y.shape}")
        _print_stats(X, y)

    return X, y


# =============================================================================
# Also save trajectory-level data (needed for rollout evaluation)
# =============================================================================

def generate_trajectories_for_eval(
    n_trajectories : int  = 200,
    random_seed    : int  = 999,
    data_dir       : str  = None,
    verbose        : bool = True,
) -> list:
    """
    Generate and save a separate set of complete trajectories (not flattened),
    intended for the autoregressive rollout evaluation.

    Each trajectory is a dict with keys: 't', 'states', 'forcing', 'y0', 'params'.
    Saved as data/eval_trajectories.npy (list of dicts).

    Returns
    -------
    trajectories : list of dicts
    """
    cfg      = DATA_GEN
    data_dir = data_dir or cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    trajectories = []

    iterator = tqdm(range(n_trajectories), desc="Generating eval trajectories") \
               if verbose else range(n_trajectories)

    for _ in iterator:
        y0     = sample_initial_condition(rng, cfg)
        params = perturb_params(ODE_PARAMS, rng, cfg)
        result = run_npzd(y0, params=params, solver_cfg=SOLVER)

        if not result["success"]:
            continue
        if np.any(result["states"] > 1e4) or np.any(np.isnan(result["states"])):
            continue

        result["y0"]     = y0
        result["params"] = params
        trajectories.append(result)

    out_path = os.path.join(data_dir, "eval_trajectories.npy")
    np.save(out_path, trajectories, allow_pickle=True)

    if verbose:
        print(f"\nEval trajectories saved: {len(trajectories)} -> {out_path}")

    return trajectories


# =============================================================================
# Utilities
# =============================================================================

def _print_stats(X: np.ndarray, y: np.ndarray) -> None:
    feature_names = ["N", "P", "Z", "D", "I", "T"]
    target_names  = ["N'", "P'", "Z'", "D'"]

    print("\n--- Input feature statistics ---")
    for i, name in enumerate(feature_names):
        print(f"  {name:5s}  mean={X[:,i].mean():.3f}  std={X[:,i].std():.3f}  "
              f"min={X[:,i].min():.3f}  max={X[:,i].max():.3f}")

    print("\n--- Target statistics ---")
    for i, name in enumerate(target_names):
        print(f"  {name:5s}  mean={y[:,i].mean():.3f}  std={y[:,i].std():.3f}  "
              f"min={y[:,i].min():.3f}  max={y[:,i].max():.3f}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    X, y = generate_dataset(verbose=True)
    generate_trajectories_for_eval(verbose=True)