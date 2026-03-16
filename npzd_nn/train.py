# =============================================================================
# train.py
# Training loop with:
#   - MSE loss on normalised increments  Δs
#   - N conservation auxiliary loss:  (ΔN + ΔP + ΔZ + ΔD)² ≈ 0
#   - Gaussian input noise on state features [N,P,Z,D] during training
#   - Adam + ReduceLROnPlateau scheduler
#   - Early stopping and best-checkpoint saving
#   - Per-variable MAE logging (physical units)
#   - Loss curve figure with conservation loss panel
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import TRAIN
from model import build_model
from dataset import build_dataloaders, Normaliser


# =============================================================================
# Per-variable MAE in physical units
# =============================================================================

def compute_per_var_mae(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
    norm       : Normaliser,
    device     : str,
) -> np.ndarray:
    """MAE of increments [mmol N m⁻³] per variable on one DataLoader."""
    model.eval()
    preds_list, tgts_list = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = model(Xb.to(device)).cpu().numpy()
            preds_list.append(p)
            tgts_list.append(yb.numpy())
    preds = np.concatenate(preds_list, axis=0)
    tgts  = np.concatenate(tgts_list,  axis=0)
    # Both in normalised space; inverse-transform to physical
    preds_phys = norm.inverse_transform_y(preds)
    tgts_phys  = norm.inverse_transform_y(tgts)
    return np.abs(preds_phys - tgts_phys).mean(axis=0)


# =============================================================================
# Conservation auxiliary loss (delta formulation)
# In the delta formulation N conservation simply means sum(Δs) = 0.
# We penalise (ΔN + ΔP + ΔZ + ΔD)² in physical units.
# =============================================================================

def conservation_loss_fn(
    pred_norm : torch.Tensor,
    norm      : Normaliser,
    device    : str,
) -> torch.Tensor:
    """
    Mean squared sum of predicted increments.
    Physically this should equal zero (N conservation).

    pred_norm : (batch, 4) normalised predictions [ΔN, ΔP, ΔZ, ΔD]
    """
    std_y  = torch.tensor(norm.std_y,  dtype=torch.float32, device=device)
    mean_y = torch.tensor(norm.mean_y, dtype=torch.float32, device=device)
    pred_phys = pred_norm * std_y + mean_y       # (batch, 4) physical
    sum_delta = pred_phys.sum(dim=1)              # (batch,)
    return (sum_delta ** 2).mean()


# =============================================================================
# Main training loop
# =============================================================================

def train(
    train_loader : torch.utils.data.DataLoader = None,
    val_loader   : torch.utils.data.DataLoader = None,
    norm         : Normaliser = None,
    cfg          : dict = None,
    verbose      : bool = True,
) -> tuple:

    cfg    = cfg or TRAIN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device        : {device}")

    torch.manual_seed(cfg["random_seed"])

    if train_loader is None or val_loader is None or norm is None:
        train_loader, val_loader, _, norm = build_dataloaders(verbose=verbose)

    model = build_model(device=device)
    if verbose:
        print(f"Parameters    : {model.count_parameters():,}")

    cons_w    = cfg.get("conservation_weight", 0.0)
    noise_std = cfg.get("input_noise_std", 0.0)
    criterion = nn.MSELoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min",
        patience=cfg["lr_patience"],
        factor=cfg["lr_factor"],
        min_lr=cfg["lr_min"],
    )
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    history = {"train_mse": [], "train_cons": [], "val_mse": [], "lr": []}
    best_val   = float("inf")
    patience_c = 0
    best_epoch = 0

    if verbose:
        print(f"Conservation w: {cons_w}")
        print(f"Input noise σ : {noise_std}\n")
        hdr = (f"{'Ep':>5}  {'TrMSE':>10}  {'Cons':>10}  "
               f"{'ValMSE':>10}  {'LR':>8}  "
               f"{'N MAE':>7}  {'P MAE':>7}  {'Z MAE':>7}  {'D MAE':>7}")
        print(hdr)
        print("-" * len(hdr))

    t0 = time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):

        # ---- Train ----
        model.train()
        tr_mse_sum = tr_cons_sum = 0.0
        n_batches  = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            if noise_std > 0.0:
                noise = torch.randn_like(Xb[:, :4]) * noise_std
                Xb_in = Xb.clone()
                Xb_in[:, :4] = Xb[:, :4] + noise
            else:
                Xb_in = Xb

            optimiser.zero_grad()
            pred = model(Xb_in)
            mse  = criterion(pred, yb)

            if cons_w > 0.0:
                cons = conservation_loss_fn(pred, norm, device)
                loss = mse + cons_w * cons
                tr_cons_sum += cons.item()
            else:
                loss = mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()

            tr_mse_sum += mse.item()
            n_batches  += 1

        tr_mse  = tr_mse_sum  / n_batches
        tr_cons = tr_cons_sum / n_batches

        # ---- Validate ----
        model.eval()
        va_mse_sum = 0.0
        n_vb = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                va_mse_sum += criterion(model(Xb.to(device)), yb.to(device)).item()
                n_vb += 1
        va_mse = va_mse_sum / n_vb
        lr_now = optimiser.param_groups[0]["lr"]
        scheduler.step(va_mse)

        history["train_mse"].append(tr_mse)
        history["train_cons"].append(tr_cons)
        history["val_mse"].append(va_mse)
        history["lr"].append(lr_now)

        # Checkpoint
        if va_mse < best_val:
            best_val   = va_mse
            best_epoch = epoch
            patience_c = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_mse": va_mse, "train_mse": tr_mse},
                       cfg["best_model_file"])
        else:
            patience_c += 1

        if verbose and (epoch % cfg["log_every_n_epochs"] == 0 or epoch == 1):
            mae = compute_per_var_mae(model, val_loader, norm, device)
            print(f"{epoch:>5}  {tr_mse:>10.6f}  {tr_cons:>10.6f}  "
                  f"{va_mse:>10.6f}  {lr_now:>8.1e}  "
                  f"{mae[0]:>7.4f}  {mae[1]:>7.4f}  "
                  f"{mae[2]:>7.4f}  {mae[3]:>7.4f}")

        if patience_c >= cfg["early_stop_patience"]:
            if verbose:
                print(f"\nEarly stop at epoch {epoch}  "
                      f"(best val MSE {best_val:.6f} at epoch {best_epoch})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTraining time : {elapsed:.1f}s")
        print(f"Best val MSE  : {best_val:.6f}  (epoch {best_epoch})")

    # Reload best checkpoint
    ckpt = torch.load(cfg["best_model_file"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _plot_loss_curve(history, cfg)
    return model, history


# =============================================================================
# Loss curve
# =============================================================================

def _plot_loss_curve(history: dict, cfg: dict) -> None:
    has_cons = any(v > 0 for v in history["train_cons"])
    rows = 3 if has_cons else 2
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3 * rows), sharex=True)
    eps = range(1, len(history["train_mse"]) + 1)

    axes[0].plot(eps, history["train_mse"], label="Train MSE", color="steelblue")
    axes[0].plot(eps, history["val_mse"],   label="Val MSE",   color="tomato")
    axes[0].set_ylabel("MSE (normalised Δs)")
    axes[0].set_title("Training History")
    axes[0].legend(); axes[0].set_yscale("log"); axes[0].grid(alpha=0.3)

    if has_cons:
        axes[1].plot(eps, history["train_cons"], color="mediumseagreen",
                     label="N Conservation Loss")
        axes[1].set_ylabel("Cons. MSE  [(mmol N m⁻³)²]")
        axes[1].legend(); axes[1].set_yscale("log"); axes[1].grid(alpha=0.3)

    axes[-1].plot(eps, history["lr"], color="darkorange", label="LR")
    axes[-1].set_ylabel("Learning Rate")
    axes[-1].set_xlabel("Epoch")
    axes[-1].legend(); axes[-1].set_yscale("log"); axes[-1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", "loss_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Loss curve → {path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    train(verbose=True)
