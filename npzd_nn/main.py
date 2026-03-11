# =============================================================================
# main.py
# Orchestrates the full NPZD neural network experiment end-to-end.
#
# Pipeline stages (all can be individually skipped via flags):
#   1. Data generation  — run NPZD ODE Monte Carlo, save X.npy, y.npy
#   2. Training         — train MLP, save best checkpoint
#   3. Evaluation       — autoregressive rollout on held-out trajectories
#
# Usage:
#   python main.py                         # run full pipeline
#   python main.py --skip-datagen          # skip data generation (use existing)
#   python main.py --skip-datagen --skip-train  # evaluation only
#   python main.py --n-traj 1000           # faster run with fewer trajectories
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="NPZD Feedforward Neural Network Experiment"
    )
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip data generation (requires existing data/X.npy, data/y.npy)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (requires existing checkpoint)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="Override number of ODE trajectories (default: from config)")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Override number of rollout trajectories in evaluation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max training epochs")
    return parser.parse_args()


def banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    args = parse_args()

    total_start = time.time()

    # -----------------------------------------------------------------------
    # Apply CLI overrides to config
    # -----------------------------------------------------------------------
    import config as cfg_module

    if args.n_traj is not None:
        cfg_module.DATA_GEN["n_trajectories"] = args.n_traj
        print(f"[override] n_trajectories = {args.n_traj}")

    if args.epochs is not None:
        cfg_module.TRAIN["max_epochs"] = args.epochs
        print(f"[override] max_epochs = {args.epochs}")

    # -----------------------------------------------------------------------
    # STAGE 1 — Data Generation
    # -----------------------------------------------------------------------
    banner("STAGE 1 — Data Generation")

    if args.skip_datagen:
        print("Skipping data generation (--skip-datagen flag set).")
        # Verify files exist
        for f in [cfg_module.DATA_GEN["X_file"], cfg_module.DATA_GEN["y_file"]]:
            if not os.path.exists(f):
                print(f"ERROR: Required file not found: {f}")
                sys.exit(1)
        print(f"Using existing: {cfg_module.DATA_GEN['X_file']}, "
              f"{cfg_module.DATA_GEN['y_file']}")
    else:
        from data_generator import generate_dataset, generate_trajectories_for_eval

        t0 = time.time()
        X, y = generate_dataset(verbose=True)
        print(f"\nDataset generation time: {time.time()-t0:.1f}s")

        t0 = time.time()
        generate_trajectories_for_eval(n_trajectories=200, verbose=True)
        print(f"Eval trajectory generation time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # STAGE 2 — Build DataLoaders (always needed for training + final eval)
    # -----------------------------------------------------------------------
    banner("STAGE 2 — Building DataLoaders")

    from dataset import build_dataloaders
    train_loader, val_loader, test_loader, normaliser = build_dataloaders(verbose=True)

    # -----------------------------------------------------------------------
    # STAGE 3 — Training
    # -----------------------------------------------------------------------
    banner("STAGE 3 — Training")

    if args.skip_train:
        print("Skipping training (--skip-train flag set).")
        ckpt = cfg_module.TRAIN["best_model_file"]
        if not os.path.exists(ckpt):
            print(f"ERROR: Checkpoint not found: {ckpt}")
            sys.exit(1)
        print(f"Using existing checkpoint: {ckpt}")
    else:
        from train import train

        t0 = time.time()
        model, history = train(
            train_loader=train_loader,
            val_loader=val_loader,
            normaliser=normaliser,
            verbose=True,
        )
        print(f"\nTraining time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # STAGE 4 — Final test-set MSE
    # -----------------------------------------------------------------------
    banner("STAGE 4 — Test Set Evaluation")

    import torch
    import torch.nn as nn
    from model import build_model
    from train import compute_per_var_mae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_eval = build_model(device=device)
    ckpt = torch.load(cfg_module.TRAIN["best_model_file"], map_location=device)
    model_eval.load_state_dict(ckpt["model_state"])
    model_eval.eval()

    criterion = nn.MSELoss()
    test_loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred    = model_eval(X_batch)
            test_loss_sum += criterion(pred, y_batch).item()
            n_batches     += 1

    test_mse = test_loss_sum / n_batches
    test_mae = compute_per_var_mae(model_eval, test_loader, normaliser, device)

    print(f"\nTest MSE (normalised):  {test_mse:.6f}")
    print(f"Test MAE per variable [mmol N m^-3]:")
    for name, mae in zip(["N", "P", "Z", "D"], test_mae):
        print(f"  {name}: {mae:.4f}")

    # -----------------------------------------------------------------------
    # STAGE 5 — Autoregressive Rollout Evaluation
    # -----------------------------------------------------------------------
    banner("STAGE 5 — Autoregressive Rollout")

    if args.skip_eval:
        print("Skipping rollout evaluation (--skip-eval flag set).")
    else:
        from evaluate import evaluate

        n_eval = args.n_eval or cfg_module.EVAL["n_rollout_trajectories"]
        all_metrics = evaluate(n_trajectories=n_eval, verbose=True)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    banner("EXPERIMENT COMPLETE")
    print(f"Total wall time: {time.time()-total_start:.1f}s")
    print(f"\nOutputs:")
    print(f"  Data           -> {cfg_module.DATA_GEN['data_dir']}/")
    print(f"  Checkpoints    -> {cfg_module.TRAIN['checkpoint_dir']}/")
    print(f"  Figures        -> {cfg_module.EVAL['figures_dir']}/")


if __name__ == "__main__":
    main()