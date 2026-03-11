# =============================================================================
# config.py
# Central configuration file for the NPZD Neural Network Experiment.
# All parameters are defined here. Import this module everywhere else.
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
# NPZD ODE — Ecological Parameters (nominal values)
# -----------------------------------------------------------------------------
ODE_PARAMS = {
    # Phytoplankton growth
    "Vm_a"  : 0.6,    # [d^-1] Max growth rate coefficient
    "Vm_b"  : 1.066,  # [-]    Temperature factor base (doubling ~10°C)
    "kN"    : 0.5,    # [mmol N m^-3] Half-saturation for nutrient uptake
    "kI"    : 30.0,   # [W m^-2]      Half-saturation for light

    # Zooplankton grazing
    "Rm"    : 1.0,    # [d^-1]              Max grazing rate
    "lam"   : 0.1,    # [(mmol N m^-3)^-1]  Ivlev grazing constant

    # Partitioning of grazing products
    "alpha" : 0.1,    # [-] Fraction of grazing excreted as dissolved N (back to N)
    "beta"  : 0.6,    # [-] Fraction of grazing assimilated into zooplankton biomass
    # (1 - alpha - beta) => 0.3 goes to detritus

    # Loss rates
    "eps"   : 0.05,   # [d^-1] Phytoplankton excretion/mortality rate
    "g"     : 0.20,   # [d^-1] Zooplankton mortality rate
    "phi"   : 0.10,   # [d^-1] Detritus remineralization rate
}

# -----------------------------------------------------------------------------
# Environmental Forcing — Seasonal Cycle
# -----------------------------------------------------------------------------
FORCING = {
    # PAR (Photosynthetically Active Radiation) — sinusoidal annual cycle
    "I_mean"  : 80.0,   # [W m^-2] Annual mean PAR
    "I_amp"   : 60.0,   # [W m^-2] Seasonal amplitude
    "I_phase" : 0.0,    # [rad]    Phase offset (0 = max at day 91, ~spring equinox)

    # Sea Surface Temperature — sinusoidal annual cycle
    "T_mean"  : 10.0,   # [°C] Annual mean SST
    "T_amp"   :  6.0,   # [°C] Seasonal amplitude
    "T_phase" : 0.5,    # [rad] Phase offset (temperature lags light slightly)
}

# -----------------------------------------------------------------------------
# ODE Solver Settings
# -----------------------------------------------------------------------------
SOLVER = {
    "t_start"   : 0,     # [days]
    "t_end"     : 365,   # [days] One full year
    "n_steps"   : 365,   # Number of daily output points
    "method"    : "RK45",
    "rtol"      : 1e-6,
    "atol"      : 1e-8,
}

# -----------------------------------------------------------------------------
# Data Generation — Monte Carlo Sampling
# -----------------------------------------------------------------------------
DATA_GEN = {
    "n_trajectories"    : 5000,   # Number of ODE runs (trajectories)
    "random_seed"       : 42,

    # Initial condition sampling ranges [min, max] in mmol N m^-3
    "N0_range"  : (0.5,  10.0),
    "P0_range"  : (0.05,  3.0),
    "Z0_range"  : (0.02,  1.5),
    "D0_range"  : (0.0,   2.0),

    # Parameter perturbation: fraction of nominal value (+/- this fraction)
    # Set to 0.0 to keep parameters fixed across all trajectories
    "param_perturb_frac": 0.25,

    # Which parameters to perturb (subset of ODE_PARAMS keys)
    "perturb_params": ["Vm_a", "Rm", "kN", "eps", "phi"],

    # Max samples to keep after extracting all one-step pairs
    "max_samples": 200_000,

    # Output file paths
    "data_dir"  : "data",
    "X_file"    : "data/X.npy",
    "y_file"    : "data/y.npy",
}

# -----------------------------------------------------------------------------
# Dataset / Normalisation
# -----------------------------------------------------------------------------
DATASET = {
    # Fraction of trajectories for each split (split at trajectory level)
    "train_frac" : 0.70,
    "val_frac"   : 0.15,
    "test_frac"  : 0.15,

    # Input feature names (for logging/plotting)
    "feature_names" : ["N", "P", "Z", "D", "I", "T"],
    # Output target names
    "target_names"  : ["N", "P", "Z", "D"],
}

# -----------------------------------------------------------------------------
# Neural Network Architecture
# -----------------------------------------------------------------------------
MODEL = {
    "input_dim"   : 6,          # N, P, Z, D, I (light), T (temperature)
    "output_dim"  : 4,          # N, P, Z, D at next timestep
    "hidden_dims" : [128, 128, 64],  # Neurons per hidden layer
    "activation"  : "relu",
    # Enforce non-negativity of outputs (concentrations >= 0)
    "output_clamp": True,
    "clamp_min"   : 0.0,
}

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
TRAIN = {
    "batch_size"    : 512,
    "max_epochs"    : 200,
    "learning_rate" : 1e-3,
    "weight_decay"  : 1e-5,     # L2 regularisation

    # ReduceLROnPlateau scheduler
    "lr_patience"   : 10,
    "lr_factor"     : 0.5,
    "lr_min"        : 1e-6,

    # Early stopping
    "early_stop_patience" : 25,

    # Checkpoint
    "checkpoint_dir": "checkpoints",
    "best_model_file": "checkpoints/best_model.pt",

    # Logging
    "log_every_n_epochs": 5,

    "random_seed"   : 0,
}

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
EVAL = {
    "n_rollout_trajectories" : 100,   # How many held-out trajectories to roll out
    "figures_dir"            : "figures",
}