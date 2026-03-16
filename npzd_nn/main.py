# =============================================================================
# main.py  —  End-to-end pipeline orchestrator.
#
# Stages:
#   1. Data generation   — Monte Carlo NPZD ODE runs → X.npy, y.npy
#   2. DataLoaders       — normalisation + train/val/test split
#   3. Training          — MLP training with conservation loss
#   4. Test-set MSE      — single-step accuracy on held-out pairs
#   5. Rollout eval      — autoregressive evaluation on full trajectories
#
# Usage:
#   python main.py                            # full pipeline
#   python main.py --skip-datagen             # reuse existing data
#   python main.py --skip-datagen --skip-train  # eval only
#   python main.py --n-traj 500 --epochs 30  # quick smoke test
# =============================================================================

import os
import sys
import time
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="NPZD Neural Network Experiment")
    p.add_argument("--skip-datagen",  action="store_true")
    p.add_argument("--skip-train",    action="store_true")
    p.add_argument("--skip-eval",     action="store_true")
    p.add_argument("--n-traj",   type=int, default=None,
                   help="Number of ODE trajectories for training data")
    p.add_argument("--n-eval",   type=int, default=None,
                   help="Number of rollout trajectories for evaluation")
    p.add_argument("--epochs",   type=int, default=None,
                   help="Max training epochs")
    return p.parse_args()


def banner(msg: str) -> None:
    w = 68
    print(f"\n{'='*w}\n  {msg}\n{'='*w}")


def main():
    args  = parse_args()
    t_all = time.time()

    # Apply CLI overrides before any config import side-effects
    import config as cfg_mod
    if args.n_traj  is not None: cfg_mod.DATA_GEN["n_trajectories"]  = args.n_traj
    if args.epochs  is not None: cfg_mod.TRAIN["max_epochs"]         = args.epochs
    if args.n_eval  is not None: cfg_mod.EVAL["n_rollout_plots"]     = args.n_eval

    # -----------------------------------------------------------------------
    banner("STAGE 1 — Data Generation")
    # -----------------------------------------------------------------------
    if args.skip_datagen:
        for f in [cfg_mod.DATA_GEN["X_file"], cfg_mod.DATA_GEN["y_file"]]:
            if not os.path.exists(f):
                print(f"ERROR: missing {f}"); sys.exit(1)
        print("Skipped (using existing files).")
    else:
        from data_generator import generate_dataset, generate_trajectories_for_eval
        t0 = time.time()
        generate_dataset(verbose=True)
        print(f"Done in {time.time()-t0:.1f}s")
        t0 = time.time()
        generate_trajectories_for_eval(verbose=True)
        print(f"Eval trajectories done in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    banner("STAGE 2 — DataLoaders")
    # -----------------------------------------------------------------------
    from dataset import build_dataloaders
    train_loader, val_loader, test_loader, norm = build_dataloaders(verbose=True)

    # -----------------------------------------------------------------------
    banner("STAGE 3 — Training")
    # -----------------------------------------------------------------------
    if args.skip_train:
        ckpt = cfg_mod.TRAIN["best_model_file"]
        if not os.path.exists(ckpt):
            print(f"ERROR: checkpoint not found: {ckpt}"); sys.exit(1)
        print("Skipped (using existing checkpoint).")
    else:
        from train import train
        t0 = time.time()
        train(train_loader=train_loader, val_loader=val_loader,
              norm=norm, verbose=True)
        print(f"Done in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    banner("STAGE 4 — Test Set Evaluation (single-step)")
    # -----------------------------------------------------------------------
    import torch, torch.nn as nn
    from model import build_model
    from train import compute_per_var_mae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(device=device)
    ckpt   = torch.load(cfg_mod.TRAIN["best_model_file"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    crit = nn.MSELoss()
    ts, nb = 0.0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            ts += crit(model(Xb.to(device)), yb.to(device)).item()
            nb += 1
    test_mse = ts / nb
    test_mae = compute_per_var_mae(model, test_loader, norm, device)

    print(f"\nTest MSE (normalised Δs) : {test_mse:.6f}")
    print(f"Test MAE per variable [mmol N m⁻³]:")
    for name, mae_v in zip(["N", "P", "Z", "D"], test_mae):
        print(f"  Δ{name} : {mae_v:.5f}")

    # -----------------------------------------------------------------------
    banner("STAGE 5 — Autoregressive Rollout")
    # -----------------------------------------------------------------------
    if args.skip_eval:
        print("Skipped.")
    else:
        from evaluate import evaluate
        t0 = time.time()
        evaluate(n_rollouts=cfg_mod.EVAL["n_rollout_plots"], verbose=True)
        print(f"Done in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    banner("DONE")
    # -----------------------------------------------------------------------
    print(f"Total wall time : {time.time()-t_all:.1f}s")
    print(f"Data            : {cfg_mod.DATA_GEN['data_dir']}/")
    print(f"Checkpoints     : {cfg_mod.TRAIN['checkpoint_dir']}/")
    print(f"Figures         : {cfg_mod.EVAL['figures_dir']}/")


if __name__ == "__main__":
    main()
