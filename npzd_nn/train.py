# =============================================================================
# train.py
# Training loop for the NPZD feedforward neural network.
#
# Features:
#   - MSE loss (on normalised targets)
#   - Adam optimiser with ReduceLROnPlateau scheduler
#   - Early stopping on validation loss
#   - Best model checkpointing
#   - Per-variable MAE tracking on validation set
#   - Loss curve saved as a figure
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import TRAIN, MODEL, DATASET
from model import build_model
from dataset import build_dataloaders, Normaliser


# =============================================================================
# Utility: compute per-variable MAE in physical (de-normalised) units
# =============================================================================

def compute_per_var_mae(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
    normaliser : Normaliser,
    device     : str,
) -> np.ndarray:
    """
    Evaluate per-variable MAE in physical units [mmol N m^-3] on a DataLoader.
    Returns array of shape (4,) — one value per [N, P, Z, D].
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred_norm = model(X_batch).cpu().numpy()
            all_preds.append(pred_norm)
            all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Inverse normalise both to physical space
    preds_phys   = normaliser.inverse_transform_y(preds)
    targets_phys = normaliser.inverse_transform_y(targets)

    # Clamp predictions to non-negative (physical constraint)
    preds_phys = np.clip(preds_phys, 0.0, None)

    mae_per_var = np.abs(preds_phys - targets_phys).mean(axis=0)
    return mae_per_var


# =============================================================================
# Main training function
# =============================================================================

def train(
    train_loader : torch.utils.data.DataLoader = None,
    val_loader   : torch.utils.data.DataLoader = None,
    normaliser   : Normaliser = None,
    cfg          : dict = None,
    verbose      : bool = True,
) -> tuple:
    """
    Full training loop.

    Returns
    -------
    model     : trained NPZDMLP (best checkpoint)
    history   : dict with 'train_loss', 'val_loss', 'lr' lists
    """
    cfg = cfg or TRAIN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Training on: {device}")

    torch.manual_seed(cfg["random_seed"])

    # Build model
    model = build_model(device=device)
    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    # If loaders not provided, build them
    if train_loader is None or val_loader is None or normaliser is None:
        train_loader, val_loader, _, normaliser = build_dataloaders(verbose=verbose)

    # Loss, optimiser, scheduler
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        patience=cfg["lr_patience"],
        factor=cfg["lr_factor"],
        min_lr=cfg["lr_min"],
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Training state
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss   = float("inf")
    patience_counter = 0
    best_epoch      = 0

    if verbose:
        print(f"\n{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  "
              f"{'LR':>10}  {'N MAE':>8}  {'P MAE':>8}  {'Z MAE':>8}  {'D MAE':>8}")
        print("-" * 90)

    t0 = time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):

        # --- Training phase ---
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            pred  = model(X_batch)
            loss  = criterion(pred, y_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimiser.step()

            train_loss_sum += loss.item()
            n_batches      += 1

        train_loss = train_loss_sum / n_batches

        # --- Validation phase ---
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred     = model(X_batch)
                val_loss_sum += criterion(pred, y_batch).item()
                n_val_batches += 1

        val_loss = val_loss_sum / n_val_batches
        current_lr = optimiser.param_groups[0]["lr"]

        # LR scheduler step
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "train_loss" : train_loss,
            }, cfg["best_model_file"])
        else:
            patience_counter += 1

        # Logging
        if verbose and (epoch % cfg["log_every_n_epochs"] == 0 or epoch == 1):
            mae = compute_per_var_mae(model, val_loader, normaliser, device)
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
                  f"{current_lr:>10.2e}  "
                  f"{mae[0]:>8.4f}  {mae[1]:>8.4f}  {mae[2]:>8.4f}  {mae[3]:>8.4f}")

        # Early stopping
        if patience_counter >= cfg["early_stop_patience"]:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.6f} at epoch {best_epoch})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Best val MSE: {best_val_loss:.6f} at epoch {best_epoch}")

    # Reload best checkpoint
    checkpoint = torch.load(cfg["best_model_file"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # Save loss curve
    _plot_loss_curve(history, cfg)

    return model, history


# =============================================================================
# Loss curve plot
# =============================================================================

def _plot_loss_curve(history: dict, cfg: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train MSE", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val MSE",   color="tomato")
    axes[0].set_ylabel("MSE Loss (normalised)")
    axes[0].legend()
    axes[0].set_title("Training History")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["lr"], color="darkorange", label="Learning Rate")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved -> {path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    model, history = train(verbose=True)