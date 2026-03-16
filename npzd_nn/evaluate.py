# =============================================================================
# evaluate.py
# Autoregressive rollout evaluation of the trained NPZD MLP.
#
# Delta formulation rollout:
#   Δŝ_t   = NN(s_t, I_t, T_t)           (predict increment, normalised)
#   ŝ_{t+1} = max(0, s_t + denorm(Δŝ_t)) (apply increment, clamp ≥ 0)
#
# Outputs:
#   figures/loss_curve.png           (from train.py)
#   figures/eval_summary.png         (RMSE boxplots + N conservation)
#   figures/rollouts/rollout_NNN.png (every 10th trajectory)
# =============================================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import EVAL
from model import build_model, NPZDMLP
from dataset import Normaliser
from npzd_ode import get_forcing_at_times


VAR_COLORS = ["steelblue", "forestgreen", "darkorange", "saddlebrown"]
VAR_NAMES  = ["N (Nutrient)", "P (Phytoplankton)", "Z (Zooplankton)", "D (Detritus)"]
VAR_SHORT  = ["N", "P", "Z", "D"]


# =============================================================================
# Autoregressive rollout (delta formulation)
# =============================================================================

def autoregressive_rollout(
    model      : NPZDMLP,
    norm       : Normaliser,
    y0         : np.ndarray,
    t          : np.ndarray,
    device     : str = "cpu",
) -> np.ndarray:
    """
    Roll out the trained network autoregressively for len(t) steps.

    Parameters
    ----------
    y0 : initial state [N0, P0, Z0, D0] in physical units
    t  : time array of length T

    Returns
    -------
    preds : (T, 4) array in physical units
    """
    model.eval()
    forcing_all = get_forcing_at_times(t)   # (T, 2)

    preds = np.zeros((len(t), 4), dtype=np.float32)
    preds[0] = y0

    state = y0.copy().astype(np.float32)

    with torch.no_grad():
        for step in range(len(t) - 1):
            I_t = forcing_all[step, 0]
            T_t = forcing_all[step, 1]

            x_raw  = np.array([state[0], state[1], state[2], state[3],
                                I_t, T_t], dtype=np.float32).reshape(1, -1)
            x_norm = norm.transform_X(x_raw)
            x_tens = torch.from_numpy(x_norm).to(device)

            delta_norm = model(x_tens).cpu().numpy()           # (1, 4)
            delta_phys = norm.inverse_transform_y(delta_norm)  # (1, 4)

            next_state = state + delta_phys[0]
            next_state = np.clip(next_state, 0.0, None)

            state       = next_state
            preds[step + 1] = state

    return preds


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err  = pred - truth
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mae  = np.abs(err).mean(axis=0)
    truth_mean = np.where(truth.mean(axis=0) < 1e-8, 1.0, truth.mean(axis=0))
    rrmse = rmse / truth_mean

    total_pred  = pred.sum(axis=1)
    total_truth = truth.sum(axis=1)
    N_cons_err  = np.abs(total_pred - total_truth[0]).mean()

    return {
        "rmse"       : rmse,
        "mae"        : mae,
        "rrmse"      : rrmse,
        "N_cons_err" : N_cons_err,
        "total_pred" : total_pred,
        "total_truth": total_truth,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_rollout(
    t       : np.ndarray,
    pred    : np.ndarray,
    truth   : np.ndarray,
    metrics : dict,
    traj_id : int,
    out_dir : str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Full-year comparison ----
    ax = fig.add_subplot(gs[0, :])
    for i, (c, n) in enumerate(zip(VAR_COLORS, VAR_NAMES)):
        ax.plot(t, truth[:, i], color=c, lw=1.5,  label=f"{n} ODE")
        ax.plot(t, pred[:, i],  color=c, lw=1.5, linestyle="--", label=f"{n} NN")
    ax.set_title(f"Rollout trajectory {traj_id}  |  mean RMSE "
                 f"{metrics['rmse'].mean():.4f} mmol N m⁻³")
    ax.set_ylabel("mmol N m⁻³")
    ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.25)

    # ---- Spring bloom focus ----
    ax2 = fig.add_subplot(gs[1, 0])
    mask = t <= 180
    for i, (c, s) in enumerate(zip(VAR_COLORS, VAR_SHORT)):
        ax2.plot(t[mask], truth[mask, i], color=c, lw=1.5, label=f"{s} ODE")
        ax2.plot(t[mask], pred[mask, i],  color=c, lw=1.5, ls="--", label=f"{s} NN")
    ax2.set_title("Spring bloom focus (days 0–180)")
    ax2.set_xlabel("Day"); ax2.set_ylabel("mmol N m⁻³")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(alpha=0.25)

    # ---- Absolute error ----
    ax3 = fig.add_subplot(gs[1, 1])
    win = 15
    for i, (c, s) in enumerate(zip(VAR_COLORS, VAR_SHORT)):
        err_t = np.abs(pred[:, i] - truth[:, i])
        sm    = np.convolve(err_t, np.ones(win) / win, mode="valid")
        ax3.plot(t[:len(sm)], sm, color=c, label=f"|err| {s}")
    ax3.set_title(f"|Error| {win}-day rolling mean")
    ax3.set_xlabel("Day"); ax3.set_ylabel("mmol N m⁻³")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # ---- N conservation ----
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t, metrics["total_truth"], "k-",  lw=2,   label="Total N (ODE)")
    ax4.plot(t, metrics["total_pred"],  "r--", lw=1.5, label="Total N (NN)")
    ax4.set_title(f"N Conservation  |  mean drift "
                  f"{metrics['N_cons_err']:.4f} mmol N m⁻³")
    ax4.set_xlabel("Day"); ax4.set_ylabel("mmol N m⁻³")
    ax4.legend(fontsize=9); ax4.grid(alpha=0.25)

    # ---- RMSE bar chart ----
    ax5 = fig.add_subplot(gs[2, 1])
    bars = ax5.bar(VAR_SHORT, metrics["rmse"], color=VAR_COLORS,
                   edgecolor="black", linewidth=0.7)
    for bar, v in zip(bars, metrics["rmse"]):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8)
    ax5.set_title("RMSE per variable (full rollout)")
    ax5.set_ylabel("mmol N m⁻³"); ax5.grid(alpha=0.25, axis="y")

    plt.savefig(os.path.join(out_dir, f"rollout_{traj_id:03d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(all_metrics: list, figures_dir: str) -> None:
    rmses = np.array([m["rmse"]       for m in all_metrics])
    ncons = np.array([m["N_cons_err"] for m in all_metrics])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot([rmses[:, i] for i in range(4)],
                    labels=VAR_SHORT, patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"))
    axes[0].set_title("RMSE distribution across eval trajectories")
    axes[0].set_ylabel("RMSE [mmol N m⁻³]"); axes[0].grid(alpha=0.3, axis="y")

    axes[1].hist(ncons, bins=25, color="tomato", edgecolor="black", lw=0.6)
    axes[1].axvline(ncons.mean(), color="darkred", ls="--",
                    label=f"Mean {ncons.mean():.4f}")
    axes[1].set_title("N Conservation Error distribution")
    axes[1].set_xlabel("Mean |ΔN_total| [mmol N m⁻³]")
    axes[1].set_ylabel("Count"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Evaluation Summary", fontsize=13)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "eval_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure → {figures_dir}/eval_summary.png")


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate(
    model_path  : str  = None,
    norm_path   : str  = None,
    traj_path   : str  = None,
    n_rollouts  : int  = None,
    figures_dir : str  = None,
    verbose     : bool = True,
) -> list:
    """
    Load the best model and run autoregressive rollouts on held-out trajectories.
    Saves every 10th individual rollout plot; summary uses all trajectories.
    """
    cfg = EVAL
    model_path  = model_path  or "checkpoints/best_model.pt"
    norm_path   = norm_path   or "data/normaliser.npz"
    traj_path   = traj_path   or cfg["eval_traj_file"]
    n_rollouts  = n_rollouts  or cfg["n_rollout_plots"]
    figures_dir = figures_dir or cfg["figures_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = Normaliser()
    norm.load(norm_path)

    model = build_model(device=device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if verbose:
        print(f"Model loaded  (epoch {ckpt['epoch']}, "
              f"val MSE {ckpt['val_mse']:.6f})")

    trajectories = np.load(traj_path, allow_pickle=True)
    n_eval = min(n_rollouts, len(trajectories))
    rng    = np.random.default_rng(0)
    idxs   = rng.choice(len(trajectories), size=n_eval, replace=False)

    all_metrics = []
    rollouts_dir = os.path.join(figures_dir, "rollouts")

    if verbose:
        hdr = (f"{'#':>4}  {'N RMSE':>8}  {'P RMSE':>8}  "
               f"{'Z RMSE':>8}  {'D RMSE':>8}  {'N_cons':>10}")
        print(f"\n{hdr}")
        print("-" * len(hdr))

    for rank, idx in enumerate(idxs):
        traj  = trajectories[idx]
        t     = traj["t"]
        truth = traj["states"]
        y0    = truth[0]

        pred    = autoregressive_rollout(model, norm, y0, t, device)
        metrics = compute_metrics(pred, truth)
        all_metrics.append(metrics)

        if verbose:
            r = metrics["rmse"]
            print(f"{rank:>4}  {r[0]:>8.4f}  {r[1]:>8.4f}  "
                  f"{r[2]:>8.4f}  {r[3]:>8.4f}  "
                  f"{metrics['N_cons_err']:>10.4f}")

        # Save every 10th rollout plot
        if rank % 10 == 0:
            plot_rollout(t, pred, truth, metrics, rank, rollouts_dir)

    if verbose:
        rmses_all = np.array([m["rmse"]       for m in all_metrics])
        ncons_all = np.array([m["N_cons_err"] for m in all_metrics])
        print(f"\n--- Aggregate ({n_eval} trajectories) ---")
        for i, s in enumerate(VAR_SHORT):
            print(f"  {s}: RMSE {rmses_all[:,i].mean():.4f} ± "
                  f"{rmses_all[:,i].std():.4f} mmol N m⁻³")
        print(f"  N conservation: {ncons_all.mean():.4f} ± "
              f"{ncons_all.std():.4f} mmol N m⁻³")

    plot_summary(all_metrics, figures_dir)
    return all_metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    evaluate(verbose=True)
