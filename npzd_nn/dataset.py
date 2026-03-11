# =============================================================================
# dataset.py
# PyTorch Dataset wrapper for the NPZD one-step transition data.
# Handles:
#   - Z-score normalisation (fit on train, applied to all splits)
#   - Train / validation / test splitting at the sample level
#     (trajectories were already shuffled during generation, so
#      random sample-level splitting is acceptable here)
#   - Saving and loading of normalisation statistics for inference
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from config import DATASET, TRAIN, DATA_GEN


# =============================================================================
# Normaliser
# =============================================================================

class Normaliser:
    """
    Z-score normaliser: x_norm = (x - mean) / std.
    Fit on the training data, then applied to all splits.
    """

    def __init__(self):
        self.mean_X = None
        self.std_X  = None
        self.mean_y = None
        self.std_y  = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_X = X.mean(axis=0).astype(np.float32)
        self.std_X  = X.std(axis=0).astype(np.float32)
        self.mean_y = y.mean(axis=0).astype(np.float32)
        self.std_y  = y.std(axis=0).astype(np.float32)

        # Avoid divide-by-zero for any constant feature
        self.std_X  = np.where(self.std_X  < 1e-8, 1.0, self.std_X)
        self.std_y  = np.where(self.std_y  < 1e-8, 1.0, self.std_y)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_X) / self.std_X

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean_y) / self.std_y

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalised predictions back to physical units."""
        return y_norm * self.std_y + self.mean_y

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(
            path,
            mean_X=self.mean_X, std_X=self.std_X,
            mean_y=self.mean_y, std_y=self.std_y,
        )
        print(f"Normaliser saved -> {path}")

    def load(self, path: str) -> None:
        data = np.load(path)
        self.mean_X = data["mean_X"]
        self.std_X  = data["std_X"]
        self.mean_y = data["mean_y"]
        self.std_y  = data["std_y"]
        print(f"Normaliser loaded <- {path}")


# =============================================================================
# PyTorch Dataset
# =============================================================================

class NPZDDataset(Dataset):
    """
    Wraps pre-normalised X, y arrays as a PyTorch Dataset.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : (n_samples, 6)  normalised inputs
        y : (n_samples, 4)  normalised targets
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =============================================================================
# Build datasets and dataloaders
# =============================================================================

def build_dataloaders(
    X_path       : str = None,
    y_path       : str = None,
    norm_path    : str = None,
    batch_size   : int = None,
    random_seed  : int = None,
    verbose      : bool = True,
) -> tuple:
    """
    Load raw X, y arrays, normalise, split, and return DataLoaders.

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    test_loader   : DataLoader
    normaliser    : Normaliser (fitted, for inference use)
    """
    cfg_data  = DATA_GEN
    cfg_ds    = DATASET
    cfg_train = TRAIN

    X_path    = X_path    or cfg_data["X_file"]
    y_path    = y_path    or cfg_data["y_file"]
    norm_path = norm_path or os.path.join(cfg_data["data_dir"], "normaliser.npz")
    batch_size  = batch_size  or cfg_train["batch_size"]
    random_seed = random_seed or cfg_train["random_seed"]

    # Load raw data
    X_raw = np.load(X_path)
    y_raw = np.load(y_path)

    if verbose:
        print(f"Loaded X: {X_raw.shape}   y: {y_raw.shape}")

    n = len(X_raw)
    n_train = int(n * cfg_ds["train_frac"])
    n_val   = int(n * cfg_ds["val_frac"])
    n_test  = n - n_train - n_val

    # Deterministic shuffle before splitting
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(n)
    X_raw = X_raw[idx]
    y_raw = y_raw[idx]

    # Split indices
    train_idx = slice(0, n_train)
    val_idx   = slice(n_train, n_train + n_val)
    test_idx  = slice(n_train + n_val, None)

    X_train_raw, y_train_raw = X_raw[train_idx], y_raw[train_idx]
    X_val_raw,   y_val_raw   = X_raw[val_idx],   y_raw[val_idx]
    X_test_raw,  y_test_raw  = X_raw[test_idx],  y_raw[test_idx]

    # Fit normaliser on training data only
    normaliser = Normaliser()
    normaliser.fit(X_train_raw, y_train_raw)
    normaliser.save(norm_path)

    # Apply normalisation
    X_train = normaliser.transform_X(X_train_raw)
    y_train = normaliser.transform_y(y_train_raw)

    X_val   = normaliser.transform_X(X_val_raw)
    y_val   = normaliser.transform_y(y_val_raw)

    X_test  = normaliser.transform_X(X_test_raw)
    y_test  = normaliser.transform_y(y_test_raw)

    # Build datasets
    ds_train = NPZDDataset(X_train, y_train)
    ds_val   = NPZDDataset(X_val,   y_val)
    ds_test  = NPZDDataset(X_test,  y_test)

    if verbose:
        print(f"Split sizes — train: {len(ds_train):,}  "
              f"val: {len(ds_val):,}  test: {len(ds_test):,}")

    # DataLoaders
    g = torch.Generator().manual_seed(random_seed)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_loader, normaliser


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    train_loader, val_loader, test_loader, normaliser = build_dataloaders(verbose=True)

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch: X={X_batch.shape}, y={y_batch.shape}")
    print(f"X dtype: {X_batch.dtype}, y dtype: {y_batch.dtype}")
    print(f"\nNormaliser mean_X: {normaliser.mean_X}")
    print(f"Normaliser std_X:  {normaliser.std_X}")