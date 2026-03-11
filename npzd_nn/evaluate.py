# =============================================================================
# evaluate.py
# Evaluation of the trained NPZD MLP via autoregressive rollout.
#
# Core idea:
#   Given only the initial state of a held-out trajectory, the network
#   predicts each successive timestep using its own previous output.
#   This compound prediction is compared against the ODE ground truth.
#
# Outputs:
#   - RMSE per variable over the rollout
#   - Nitrogen conservation error
#   - Plots: rollout traces, RMSE over time, spring bloom focus
#   - Summary printed to stdout
# =============================================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import EVAL, DATASET, DATA_GEN
from model import build_model, NPZDMLP
from dataset import Normaliser
from npzd_ode import get_forcing_at_times


# =============================================================================
# Autoregressive rollout
# =============================================================================

def autoregressive_rollout(
    model       : NPZDMLP,
    normaliser  : Normaliser,
    y0          : np.ndarray,
    t           : np.ndarray,
    device      : str = "cpu",
) -> np.ndarray:
    """
    Roll out the network autoregressively for len(t)-1 steps.

    Parameters
    ----------
    model      : trained NPZDMLP
    normaliser : fitted Normaliser
    y0         : initial state [N0, P0, Z0, D0]  (physical units)
    t          : time array (days), length T
    device     : torch device string

    Returns
    -------
    preds : np.ndarray of shape (T, 4)  [N, P, Z, D] in physical units
            preds[0] == y0 (the given initial condition)
    """
    model.eval()

    # Precompute all forcing values
    forcing_all = get_forcing_at_times(t)   # (T, 2)  [I, T]

    preds = np.zeros((len(t), 4), dtype=np.float32)
    preds[0] = y0

    current_state = y0.copy().astype(np.float32)

    with torch.no_grad():
        for step in range(len(t) - 1):
            I_t = forcing_all[step, 0]
            T_t = forcing_all[step, 1]

            # Build raw input vector [N, P, Z, D, I, T]
            x_raw = np.array(
                [current_state[0], current_state[1],
                 current_state[2], current_state[3],
                 I_t, T_t],
                dtype=np.float32
            ).reshape(1, -1)

            # Normalise input
            x_norm = normaliser.transform_X(x_raw)
            x_tensor = torch.from_numpy(x_norm).to(device)

            # Forward pass
            y_norm_pred = model(x_tensor).cpu().numpy()          # (1, 4)

            # Inverse normalise -> physical units
            y_phys = normaliser.inverse_transform_y(y_norm_pred)  # (1, 4)

            # Enforce non-negativity
            y_phys = np.clip(y_phys, 0.0, None)

            current_state = y_phys[0]
            preds[step + 1] = current_state

    return preds


# =============================================================================
# Compute metrics for a single rollout
# =============================================================================

def rollout_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """
    Compute per-variable RMSE, MAE, and nitrogen conservation error.

    Parameters
    ----------
    pred  : (T, 4) predicted states
    truth : (T, 4) ground truth states

    Returns
    -------
    metrics : dict
    """
    err = pred - truth

    rmse = np.sqrt((err ** 2).mean(axis=0))   # (4,)
    mae  = np.abs(err).mean(axis=0)            # (4,)

    # Nitrogen conservation: total N should be constant
    total_N_pred  = pred.sum(axis=1)
    total_N_truth = truth.sum(axis=1)
    N_conservation_err = np.abs(total_N_pred - total_N_truth[0]).mean()

    # Relative RMSE (normalised by mean of ground truth)
    truth_mean = truth.mean(axis=0)
    truth_mean = np.where(truth_mean < 1e-8, 1.0, truth_mean)
    rrmse = rmse / truth_mean

    return {
        "rmse"             : rmse,
        "mae"              : mae,
        "rrmse"            : rrmse,
        "N_conservation_err": N_conservation_err,
        "total_N_pred"     : total_N_pred,
        "total_N_truth"    : total_N_truth,
    }


# =============================================================================
# Plotting
# =============================================================================

VAR_COLORS = ["steelblue", "forestgreen", "darkorange", "saddlebrown"]
VAR_NAMES  = ["N (Nutrient)", "P (Phytoplankton)", "Z (Zooplankton)", "D (Detritus)"]
VAR_SHORT  = ["N", "P", "Z", "D"]


def plot_rollout(
    t       : np.ndarray,
    pred    : np.ndarray,
    truth   : np.ndarray,
    metrics : dict,
    traj_id : int,
    figures_dir: str,
) -> None:
    """Plot a full-year rollout comparison for one trajectory."""

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Top row: all 4 variables ----
    ax_main = fig.add_subplot(gs[0, :])
    for i, (c, name) in enumerate(zip(VAR_COLORS, VAR_NAMES)):
        ax_main.plot(t, truth[:, i], color=c, linewidth=1.5, label=f"{name} (ODE)")
        ax_main.plot(t, pred[:, i],  color=c, linewidth=1.5, linestyle="--",
                     label=f"{name} (NN)")
    ax_main.set_ylabel("mmol N m⁻³")
    ax_main.set_title(f"Autoregressive Rollout — Trajectory {traj_id}  "
                      f"(mean RMSE: {metrics['rmse'].mean():.4f})")
    ax_main.legend(fontsize=7, ncol=4)
    ax_main.grid(alpha=0.25)

    # ---- Middle row left: spring bloom focus (days 0-180) ----
    ax_bloom = fig.add_subplot(gs[1, 0])
    mask = t <= 180
    for i, (c, short) in enumerate(zip(VAR_COLORS, VAR_SHORT)):
        ax_bloom.plot(t[mask], truth[mask, i], color=c, lw=1.5, label=f"{short} ODE")
        ax_bloom.plot(t[mask], pred[mask, i],  color=c, lw=1.5, ls="--",
                      label=f"{short} NN")
    ax_bloom.set_title("Spring Bloom Focus (days 0–180)")
    ax_bloom.set_xlabel("Day")
    ax_bloom.set_ylabel("mmol N m⁻³")
    ax_bloom.legend(fontsize=7, ncol=2)
    ax_bloom.grid(alpha=0.25)

    # ---- Middle row right: per-variable RMSE over time (rolling window) ----
    ax_rmse = fig.add_subplot(gs[1, 1])
    window = 15
    for i, (c, short) in enumerate(zip(VAR_COLORS, VAR_SHORT)):
        err_t = np.abs(pred[:, i] - truth[:, i])
        if len(err_t) >= window:
            err_smooth = np.convolve(err_t, np.ones(window)/window, mode="valid")
            ax_rmse.plot(t[:len(err_smooth)], err_smooth, color=c, label=f"|err| {short}")
    ax_rmse.set_title(f"Absolute Error (15-day rolling mean)")
    ax_rmse.set_xlabel("Day")
    ax_rmse.set_ylabel("mmol N m⁻³")
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(alpha=0.25)

    # ---- Bottom row left: nitrogen conservation ----
    ax_ncons = fig.add_subplot(gs[2, 0])
    ax_ncons.plot(t, metrics["total_N_truth"], color="black",
                  lw=2, label="Total N — ODE")
    ax_ncons.plot(t, metrics["total_N_pred"],  color="red",
                  lw=1.5, ls="--", label="Total N — NN")
    ax_ncons.set_title(f"N Conservation  (drift: "
                       f"{metrics['N_conservation_err']:.4f} mmol N m⁻³)")
    ax_ncons.set_xlabel("Day")
    ax_ncons.set_ylabel("mmol N m⁻³")
    ax_ncons.legend(fontsize=9)
    ax_ncons.grid(alpha=0.25)

    # ---- Bottom row right: per-variable RMSE bar chart ----
    ax_bar = fig.add_subplot(gs[2, 1])
    bars = ax_bar.bar(VAR_SHORT, metrics["rmse"], color=VAR_COLORS, edgecolor="black",
                      linewidth=0.7)
    ax_bar.set_title("RMSE per Variable (full rollout)")
    ax_bar.set_ylabel("RMSE [mmol N m⁻³]")
    for bar, v in zip(bars, metrics["rmse"]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax_bar.grid(alpha=0.25, axis="y")

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"rollout_traj{traj_id:03d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_summary(all_metrics: list, figures_dir: str) -> None:
    """Aggregate summary figure across all evaluated trajectories."""
    rmses = np.array([m["rmse"] for m in all_metrics])       # (n_traj, 4)
    ncons = np.array([m["N_conservation_err"] for m in all_metrics])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot of per-variable RMSE
    axes[0].boxplot(
        [rmses[:, i] for i in range(4)],
        labels=VAR_SHORT,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue"),
    )
    axes[0].set_title("Per-variable RMSE distribution across test trajectories")
    axes[0].set_ylabel("RMSE [mmol N m⁻³]")
    axes[0].grid(alpha=0.3, axis="y")

    # Histogram of nitrogen conservation error
    axes[1].hist(ncons, bins=20, color="tomato", edgecolor="black", linewidth=0.6)
    axes[1].axvline(ncons.mean(), color="darkred", linestyle="--",
                    label=f"Mean: {ncons.mean():.4f}")
    axes[1].set_title("N Conservation Error distribution")
    axes[1].set_xlabel("Mean |ΔN_total| [mmol N m⁻³]")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Evaluation Summary", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "eval_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure saved -> {out_path}")


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate(
    model_path   : str = None,
    norm_path    : str = None,
    eval_traj_path: str = None,
    n_trajectories: int = None,
    figures_dir  : str = None,
    verbose      : bool = True,
) -> list:
    """
    Load the best trained model and evaluate it by autoregressive rollout
    on held-out trajectories.

    Returns
    -------
    all_metrics : list of metric dicts (one per trajectory)
    """
    cfg_eval = EVAL
    cfg_data = DATA_GEN

    model_path    = model_path    or "checkpoints/best_model.pt"
    norm_path     = norm_path     or os.path.join(cfg_data["data_dir"], "normaliser.npz")
    eval_traj_path = eval_traj_path or os.path.join(cfg_data["data_dir"],
                                                      "eval_trajectories.npy")
    n_trajectories = n_trajectories or cfg_eval["n_rollout_trajectories"]
    figures_dir   = figures_dir   or cfg_eval["figures_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load normaliser
    normaliser = Normaliser()
    normaliser.load(norm_path)

    # Load model
    model = build_model(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if verbose:
        print(f"Model loaded from {model_path}  "
              f"(trained {checkpoint['epoch']} epochs, "
              f"val MSE={checkpoint['val_loss']:.6f})")

    # Load eval trajectories
    trajectories = np.load(eval_traj_path, allow_pickle=True)
    if verbose:
        print(f"Loaded {len(trajectories)} eval trajectories from {eval_traj_path}")

    # Select a subset
    n_eval = min(n_trajectories, len(trajectories))
    rng    = np.random.default_rng(0)
    idxs   = rng.choice(len(trajectories), size=n_eval, replace=False)

    all_metrics = []

    if verbose:
        print(f"\nRunning {n_eval} autoregressive rollouts...")
        print(f"\n{'Traj':>5}  {'N RMSE':>8}  {'P RMSE':>8}  "
              f"{'Z RMSE':>8}  {'D RMSE':>8}  {'N_cons_err':>12}")
        print("-" * 60)

    for traj_id, idx in enumerate(idxs):
        traj  = trajectories[idx]
        t     = traj["t"]
        truth = traj["states"]   # (T, 4)
        y0    = truth[0]

        # Rollout
        pred = autoregressive_rollout(model, normaliser, y0, t, device)

        # Metrics
        metrics = rollout_metrics(pred, truth)
        all_metrics.append(metrics)

        if verbose:
            print(f"{traj_id:>5}  "
                  f"{metrics['rmse'][0]:>8.4f}  {metrics['rmse'][1]:>8.4f}  "
                  f"{metrics['rmse'][2]:>8.4f}  {metrics['rmse'][3]:>8.4f}  "
                  f"{metrics['N_conservation_err']:>12.4f}")

        # Individual rollout plot
        rollouts_dir = os.path.join(figures_dir, "rollouts")
        if traj_id % 10 == 0:
            rollouts_dir = os.path.join(figures_dir, "rollouts")
            plot_rollout(t, pred, truth, metrics, traj_id, rollouts_dir)

    # Aggregate summary
    if verbose:
        rmses_all = np.array([m["rmse"] for m in all_metrics])
        ncons_all = np.array([m["N_conservation_err"] for m in all_metrics])
        print(f"\n--- Aggregate across {n_eval} trajectories ---")
        for i, name in enumerate(VAR_SHORT):
            print(f"  {name}: mean RMSE = {rmses_all[:,i].mean():.4f} ± "
                  f"{rmses_all[:,i].std():.4f} mmol N m^-3")
        print(f"  N conservation error: {ncons_all.mean():.4f} ± "
              f"{ncons_all.std():.4f} mmol N m^-3")

    # Summary figure
    plot_summary(all_metrics, figures_dir)

    return all_metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    evaluate(verbose=True)