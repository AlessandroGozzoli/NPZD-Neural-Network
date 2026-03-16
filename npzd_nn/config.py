# =============================================================================
# config.py  —  Central parameter store.
# Every number in the project lives here. Import from this file everywhere.
# =============================================================================

# -----------------------------------------------------------------------------
# NPZD ecological parameters  (nominal / mean values)
# -----------------------------------------------------------------------------
ODE_PARAMS = {
    # Phytoplankton growth  (Eppley 1972 temperature function)
    "Vm_a" : 0.6,    # [d⁻¹]  max growth rate coefficient
    "Vm_b" : 1.066,  # [—]    temperature base (≈ doubling per 10 °C)
    "kN"   : 0.5,    # [mmol N m⁻³]  Michaelis-Menten half-sat. for N
    "kI"   : 30.0,   # [W m⁻²]       Michaelis-Menten half-sat. for light

    # Zooplankton grazing  (Ivlev 1955)
    "Rm"   : 0.8,    # [d⁻¹]                   max grazing rate
    "lam"  : 0.06,   # [(mmol N m⁻³)⁻¹]        Ivlev saturation constant

    # Grazing partitioning  (must satisfy: alpha + beta < 1)
    "alpha": 0.10,   # [—]  fraction excreted back to dissolved N
    "beta" : 0.60,   # [—]  fraction assimilated into zooplankton biomass
    # Remainder (1 - alpha - beta) = 0.30  →  detritus (egestion)

    # Loss rates
    "eps"  : 0.03,   # [d⁻¹]  phytoplankton specific mortality/excretion
    "g"    : 0.15,   # [d⁻¹]  zooplankton specific mortality
    "phi"  : 0.07,   # [d⁻¹]  detritus remineralisation rate
}

# -----------------------------------------------------------------------------
# Environmental forcing — sinusoidal annual cycle
# -----------------------------------------------------------------------------
FORCING = {
    # PAR (Photosynthetically Active Radiation)
    "I_mean"  : 75.0,   # [W m⁻²]  annual mean
    "I_amp"   : 55.0,   # [W m⁻²]  seasonal amplitude
    "I_phase" : 0.0,    # [rad]     0 → max at day ~91 (spring equinox)

    # Sea Surface Temperature
    "T_mean"  : 10.0,   # [°C]   annual mean
    "T_amp"   :  6.0,   # [°C]   seasonal amplitude
    "T_phase" :  0.5,   # [rad]  slight lag vs. light (ocean thermal inertia)

    # Seasonal nutrient supply via winter deep-water mixing.
    # This is the standard mechanism that drives the spring bloom in
    # 0D NPZD models (Fasham et al. 1990). Without it, inorganic nitrogen
    # is permanently depleted and the system collapses to a dead attractor.
    #
    # Physics: in winter, the mixed layer deepens and entrains nutrient-rich
    # deep water. We parameterise this as a restoring term in dN/dt:
    #   dN/dt += kappa(t) * (N_deep - N)
    # where kappa(t) = kappa_max * max(0, cos(2π*t/365))
    # which peaks on Jan 1 and is zero from spring equinox to autumn equinox.
    #
    # Note: this makes the system OPEN — total N is not conserved.
    "kappa_max" : 0.08,   # [d⁻¹]        max winter mixing rate
    "N_deep"    : 8.0,    # [mmol N m⁻³] deep water N concentration
}

# -----------------------------------------------------------------------------
# ODE solver  —  Radau (implicit, order 5, A-stable; designed for stiff ODEs)
# -----------------------------------------------------------------------------
SOLVER = {
    "method"      : "Radau",
    "t_start"     : 0,
    "t_end"       : 365,
    "n_steps"     : 366,     # daily snapshots: t = 0, 1, …, 365
    "rtol"        : 1e-8,
    "atol"        : 1e-10,
    # No spin-up: initial conditions represent winter (high N, low biology),
    # which is the natural starting state. Spin-up drove the system to a
    # degenerate nutrient-depleted fixed point.
    "spinup_days" : 0,
}

# -----------------------------------------------------------------------------
# Data generation — Monte Carlo
# -----------------------------------------------------------------------------
DATA_GEN = {
    "n_trajectories"      : 5000,
    "random_seed"         : 42,

    # Winter initial conditions: high inorganic N, low biology.
    # The bloom emerges naturally from these conditions as light increases.
    # Each variable is sampled independently (system is open, so no N budget
    # constraint is needed).
    "N0_range" : (4.0, 12.0),   # [mmol N m⁻³]  high winter nutrients
    "P0_range" : (0.01, 0.3),   # [mmol N m⁻³]  low winter phytoplankton
    "Z0_range" : (0.01, 0.15),  # [mmol N m⁻³]  low winter zooplankton
    "D0_range" : (0.01, 0.5),   # [mmol N m⁻³]  modest detritus

    # Ecological parameter perturbation (±frac around nominal values)
    "param_perturb_frac" : 0.20,
    "perturb_params"     : ["Vm_a", "Rm", "kN", "eps", "phi", "lam"],

    # Cap on one-step pairs kept for training
    "max_samples"         : 200_000,

    # Post-solve quality filters
    "max_N_drift_frac"    : 0.50,   # generous: system is open, some drift expected
    "min_state_value"     : -1e-6,
    "max_concentration"   : 40.0,

    # Paths
    "data_dir"  : "data",
    "X_file"    : "data/X.npy",
    "y_file"    : "data/y.npy",
}

# -----------------------------------------------------------------------------
# Dataset / normalisation
# -----------------------------------------------------------------------------
DATASET = {
    "train_frac"    : 0.70,
    "val_frac"      : 0.15,
    "test_frac"     : 0.15,
    "feature_names" : ["N", "P", "Z", "D", "I", "T"],
    # Targets are STATE INCREMENTS  Δs = s_{t+1} − s_t
    "target_names"  : ["ΔN", "ΔP", "ΔZ", "ΔD"],
}

# -----------------------------------------------------------------------------
# Neural network  —  delta (increment) formulation
# The NN predicts Δs = s_{t+1} − s_t rather than s_{t+1} directly.
# Benefits: smaller signal to learn, simpler conservation constraint
# (sum(Δs) ≈ 0), better rollout stability.
# -----------------------------------------------------------------------------
MODEL = {
    "input_dim"   : 6,
    "output_dim"  : 4,
    "hidden_dims" : [128, 128, 64],
    "dropout_p"   : 0.0,
}

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
TRAIN = {
    "batch_size"          : 512,
    "max_epochs"          : 200,
    "learning_rate"       : 1e-3,
    "weight_decay"        : 1e-5,

    "lr_patience"         : 10,
    "lr_factor"           : 0.5,
    "lr_min"              : 1e-6,

    "early_stop_patience" : 25,

    # Conservation loss is DISABLED: the system is open (seasonal mixing adds
    # external N), so sum(Δs) ≠ 0 by design. A conservation penalty would
    # incorrectly punish the network for learning the mixing signal.
    "conservation_weight" : 0.0,

    # Small Gaussian noise on state inputs [N,P,Z,D] during training only.
    "input_noise_std"     : 0.005,

    "checkpoint_dir"      : "checkpoints",
    "best_model_file"     : "checkpoints/best_model.pt",
    "log_every_n_epochs"  : 5,
    "random_seed"         : 0,
}

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
EVAL = {
    "n_eval_trajectories" : 200,
    "n_rollout_plots"     : 200,
    "figures_dir"         : "figures",
    "eval_traj_file"      : "data/eval_trajectories.npy",
}
