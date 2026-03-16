# =============================================================================
# dataset.py
# PyTorch Dataset, Z-score normaliser, and DataLoader factory.
#
# Note on the delta formulation:
#   X = [N, P, Z, D, I, T]          (current state + forcing)
#   y = [ΔN, ΔP, ΔZ, ΔD]           (state increment to next step)
#
# The normaliser fits on X_train and y_train separately.
# During inference / rollout, only transform_X is needed on the input;
# the output is inverse-transformed to get physical increments.
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_GEN, DATASET, TRAIN


# =============================================================================
# Normaliser
# =============================================================================

class Normaliser:
    """Z-score normaliser.  Fitted on training data only."""

    def __init__(self):
        self.mean_X = self.std_X = None
        self.mean_y = self.std_y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_X = X.mean(axis=0).astype(np.float32)
        self.std_X  = X.std(axis=0).astype(np.float32)
        self.mean_y = y.mean(axis=0).astype(np.float32)
        self.std_y  = y.std(axis=0).astype(np.float32)
        # Guard against constant features
        self.std_X = np.where(self.std_X < 1e-8, 1.0, self.std_X)
        self.std_y = np.where(self.std_y < 1e-8, 1.0, self.std_y)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_X) / self.std_X).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean_y) / self.std_y).astype(np.float32)

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        return (y_norm * self.std_y + self.mean_y).astype(np.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(path, mean_X=self.mean_X, std_X=self.std_X,
                       mean_y=self.mean_y, std_y=self.std_y)

    def load(self, path: str) -> None:
        d = np.load(path)
        self.mean_X, self.std_X = d["mean_X"], d["std_X"]
        self.mean_y, self.std_y = d["mean_y"], d["std_y"]


# =============================================================================
# Dataset
# =============================================================================

class NPZDDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# DataLoader factory
# =============================================================================

def build_dataloaders(
    X_path      : str  = None,
    y_path      : str  = None,
    norm_path   : str  = None,
    batch_size  : int  = None,
    random_seed : int  = None,
    verbose     : bool = True,
) -> tuple:
    """
    Load raw arrays, normalise, split, return (train, val, test) DataLoaders
    and the fitted Normaliser.
    """
    cfg_d  = DATA_GEN
    cfg_ds = DATASET
    cfg_t  = TRAIN

    X_path     = X_path     or cfg_d["X_file"]
    y_path     = y_path     or cfg_d["y_file"]
    norm_path  = norm_path  or os.path.join(cfg_d["data_dir"], "normaliser.npz")
    batch_size  = batch_size  or cfg_t["batch_size"]
    random_seed = random_seed or cfg_t["random_seed"]

    X_raw = np.load(X_path).astype(np.float32)
    y_raw = np.load(y_path).astype(np.float32)

    if verbose:
        print(f"Loaded  X: {X_raw.shape}   y: {y_raw.shape}")

    n       = len(X_raw)
    n_train = int(n * cfg_ds["train_frac"])
    n_val   = int(n * cfg_ds["val_frac"])

    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(n)
    X_raw, y_raw = X_raw[idx], y_raw[idx]

    sl_tr = slice(0, n_train)
    sl_va = slice(n_train, n_train + n_val)
    sl_te = slice(n_train + n_val, None)

    norm = Normaliser()
    norm.fit(X_raw[sl_tr], y_raw[sl_tr])
    norm.save(norm_path)

    def make_ds(sl):
        return NPZDDataset(norm.transform_X(X_raw[sl]),
                           norm.transform_y(y_raw[sl]))

    ds_tr = make_ds(sl_tr)
    ds_va = make_ds(sl_va)
    ds_te = make_ds(sl_te)

    if verbose:
        print(f"Split    train={len(ds_tr):,}  val={len(ds_va):,}  test={len(ds_te):,}")

    g = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_loader, norm


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    tl, vl, tel, norm = build_dataloaders(verbose=True)
    Xb, yb = next(iter(tl))
    print(f"\nBatch shapes:  X={Xb.shape}  y={yb.shape}")
    print(f"mean_X : {norm.mean_X}")
    print(f"mean_y : {norm.mean_y}")
