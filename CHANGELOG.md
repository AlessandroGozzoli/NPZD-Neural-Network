# Changelog

---

## Version 2.0.0 — 17/03/2026

### Summary

Complete ground-up rewrite of the entire codebase, driven by four diagnosed issues in v1.0.0 (see v1.0.0 Known Issues). The primary focus of this version is the **physical and numerical correctness of the ODE system**, which is a prerequisite for all downstream components. The neural network formulation, data generation strategy, training objective, and stability filtering have all been redesigned as a consequence.

All v1.0.0 known issues have been resolved or are no longer applicable under the new architecture.

---

### `npzd_ode.py` — ODE System

#### Bug Fix: Nitrogen Double-Routing (Critical)
The original ODE contained a nitrogen mass-balance error in which the phytoplankton loss term (`P_loss = ε·P`) was simultaneously added to both `dN/dt` and `dD/dt`, while only being subtracted once from `dP/dt`. This created nitrogen from nothing on every timestep, producing total-N conservation drifts of 400–1600% and driving concentrations to blow-up. The correct routing is `P → D` only, with the return path `D → N` handled exclusively by remineralisation. The corrected equations are:

| Equation | v1.0.0 (incorrect) | v2.0.0 (correct) |
|---|---|---|
| `dN/dt` | `−PP + αGZ + εP + φD` | `−PP + αGZ + φD + Mix` |
| `dP/dt` | `+PP − GZ − εP` | `+PP − GZ − εP` *(unchanged)* |
| `dD/dt` | `(1−α−β)GZ + εP + gZ − φD` | `(1−α−β)GZ + εP + gZ − φD` *(unchanged)* |

#### New Feature: Seasonal Nutrient Mixing (Open System)
The v1.0.0 system was a closed box with no external nutrient supply. In a closed NPZD box, phytoplankton rapidly consumes all inorganic nitrogen and the system collapses to a biologically dead fixed point, no seasonal dynamics are possible. This is the physical root cause of the flat-line trajectories observed in rollout plots.

A standard seasonal mixing term has been added to `dN/dt`, following Fasham et al. (1990):

```
dN/dt += κ(t) · (N_deep − N)
κ(t)  =  κ_max · max(0, cos(2π·t/365))
```

This represents winter mixed-layer deepening entraining nutrient-rich deep water, the standard mechanism driving the spring phytoplankton bloom in all realistic 0D NPZD formulations. The system is now **open**: total N is not conserved, and seasonal bloom dynamics emerge naturally.

New parameters added to `config.FORCING`:
- `kappa_max = 0.08` [d⁻¹] — peak winter mixing rate
- `N_deep = 8.0` [mmol N m⁻³] — deep water nutrient concentration

#### New Feature: Analytical Jacobian
An analytical 4×4 Jacobian of the RHS is now provided to the solver. This eliminates finite-difference Jacobian approximation overhead and substantially reduces the number of RHS evaluations per step when using implicit solvers. The Jacobian correctly includes the `−κ(t)` contribution from the mixing term in `J[0,0]`.

#### Solver: RK45 → Radau
The explicit adaptive Runge-Kutta solver (`RK45`) has been replaced with `Radau`, an implicit Runge-Kutta method of order 5 that is A-stable and designed for stiff ODE systems. NPZD biogeochemical equations are stiff by nature (widely separated timescales between phytoplankton growth and zooplankton mortality). RK45 is not suited for stiff systems and was a contributing factor to the numerical instabilities observed in v1.0.0. Solver tolerances tightened to `rtol=1e-8`, `atol=1e-10`.

#### Spin-up Removed
A 365-day spin-up period was introduced during development as an attempted fix for transient initial behaviour. It was subsequently identified as catastrophic: it ran the system fully into the biologically dead fixed point *before* collecting any data, ensuring every trajectory started flat and stayed flat. The spin-up has been removed entirely. Initial conditions now represent winter states (high N, low biology), which is the physically correct starting point for a spring bloom simulation.

---

### `config.py` — Configuration

- Added `FORCING["kappa_max"]` and `FORCING["N_deep"]` for the mixing term.
- `SOLVER["method"]` changed from `"RK45"` to `"Radau"`.
- `SOLVER["rtol"]` tightened from `1e-6` to `1e-8`; `SOLVER["atol"]` from `1e-8` to `1e-10`.
- `SOLVER["spinup_days"]` set to `0` (spin-up removed).
- `DATA_GEN` initial condition sampling changed from N-conserving partition to independent winter ranges (see `data_generator.py` below).
- `DATASET["target_names"]` updated to `["ΔN", "ΔP", "ΔZ", "ΔD"]` to reflect the delta formulation.
- `TRAIN["conservation_weight"]` set to `0.0` (conservation loss disabled for open system).
- `TRAIN["input_noise_std"]` added (`0.005`) — small Gaussian noise on state inputs during training.
- `MODEL["dropout_p"]` added (`0.0`, disabled by default).
- `STABILITY` block removed entirely (replaced by two minimal checks in `data_generator.py`).

Default configurations for v2.0.0:

- **Data generation batch size:** 5,000 samples (random seed: 42)
- **Initial state ranges for chemical variables:**
  | Variable | Min   | Max   |
  |----------|-------|-------|
  | `N0`     | 4.00  | 12.00 |
  | `P0`     | 0.01  | 0.30  |
  | `Z0`     | 0.01  | 0.15  |
  | `D0`     | 0.01  | 0.50  |

  > Perturbation fraction of 0.20 applied to ecological parameters across trajectories.
- **Model architecture:** 6 inputs, 4 outputs, 3 hidden layers (128 → 128 → 64 nodes)
- **Training batch:** 512 samples, max 200 epochs
- **Evaluation batch:** 200 trajectories

---

### `data_generator.py` — Data Generation

#### Initial Condition Sampling Redesigned
The v1.0.0 approach sampled N, P, Z, D independently from unconstrained ranges, routinely producing states where one compartment exceeded the entire plausible nitrogen budget. An intermediate N-conserving strategy was introduced during development (sampling `total_N` then partitioning among compartments via fractional draws), but this is only meaningful for a closed system and was removed when the system became open. The final approach samples each variable independently from physically calibrated winter ranges.

#### Target Formulation: Absolute State → State Increment (Delta)
Training targets changed from `y = s_{t+1}` (absolute next state) to `y = Δs = s_{t+1} − s_t` (state increment). This makes the learning signal smaller and smoother, improves rollout stability, and simplifies the physics-informed loss formulation.

#### Stability Filters Simplified
The v1.0.0 code had an elaborate five-check stability classifier (`is_trajectory_stable`) including oscillation detection via extrema counting, maximum daily relative change, and N conservation drift. All of these were removed for two reasons: (1) they were rejecting 100% of valid trajectories, and (2) with Radau + analytical Jacobian, the solver produces clean trajectories that do not require complex post-hoc filtering. Two minimal checks remain: blow-up/NaN detection and hard negative concentration detection.

#### N Drift Filter Removed
An intermediate version added a `max_N_drift_frac` filter that rejected trajectories where total N drifted more than 50% from its initial value. This was correctly identifying that the mixing term produces N drift, then incorrectly rejecting those trajectories as unstable. Since total N drifting is correct physics for an open system, the check was removed entirely.

---

### `train.py` — Training Loop

#### Conservation Loss Disabled
A physics-informed auxiliary loss penalising `(ΔN + ΔP + ΔZ + ΔD)²` was introduced during development. This is only meaningful for a closed system where the sum of increments should be zero. With the open system (mixing adds external N), this sum is non-zero by design, the conservation loss was penalising the network for correctly learning the mixing signal. Disabled by setting `conservation_weight = 0.0`.

#### Input Noise Injection Added
Small Gaussian noise (`σ = 0.005`, normalised units) is applied to the four state inputs `[N, P, Z, D]` during training only. This prevents the network from memorising the exact training distribution and discourages convergence to spurious attractor states. Noise is not applied to forcing inputs `[I, T]`, which are deterministic functions of time.

#### Loss Curve: Conservation Panel
The loss curve figure now conditionally renders a third panel showing the conservation loss per epoch when `conservation_weight > 0`. When the conservation loss is disabled (current default), the figure reverts to the standard two-panel layout.

---

### `dataset.py` — Normalisation and DataLoaders

No structural changes. Updated internally to handle delta targets (`Δs`) instead of absolute next states. The `Normaliser` class and `build_dataloaders` factory are unchanged in interface.

---

### `model.py` — Neural Network

Minor additions only. `dropout_p` parameter added (default `0.0`, disabled). Architecture unchanged: 6→128→128→64→4 with ReLU activations and Kaiming normal initialisation.

---

### `evaluate.py` — Evaluation

#### Rollout Formula Updated for Delta Formulation
The autoregressive rollout formula updated from:
```
ŝ_{t+1} = denorm(NN(norm(s_t, I_t, T_t)))
```
to:
```
ŝ_{t+1} = max(0,  s_t  +  denorm(NN(norm(s_t, I_t, T_t))))
```
The additive correction is more stable in long rollouts because the network learns small residuals rather than absolute values.

#### Rollout Plots: Subfolder
Individual rollout PNGs are now saved to `figures/rollouts/` instead of `figures/`. The summary figure (`eval_summary.png`) remains in `figures/`.

#### Rollout Plot Frequency
Individual trajectory PNG files are saved every 10th trajectory (`rank % 10 == 0`) rather than for every evaluated trajectory, reducing disk usage for large evaluation sets.

---

### `main.py` — Orchestrator

No structural changes. Updated stage comments and banner text to reflect the new pipeline. `generate_trajectories_for_eval` now reads `n_eval_trajectories` from `config.EVAL` rather than a hardcoded value.

---

### Bug Fixes

| # | File | Description |
|---|------|-------------|
| 1 | `npzd_ode.py` | `P_loss` double-routed to both `dN` and `dD`, creating nitrogen each timestep |
| 2 | `npzd_ode.py` | `get_forcing_at_times` definition lost when `mixing_rate` was inserted via str_replace; function body was unreachable dead code inside `mixing_rate` after its `return` statement |
| 3 | `data_generator.py` | `max_N_drift_frac` filter rejected all trajectories in open system (N drift is correct physics) |
| 4 | `data_generator.py` | N-conserving IC sampler assumed closed system; replaced with independent winter ranges |

---

### Resolved Issues from v1.0.0

| v1.0.0 Issue | Resolution |
|---|---|
| **Raw Model Error (30–35 mmol N m⁻³)** | Downstream of the ODE bug and flat-line trajectories; resolved by fixing the ODE and adding the mixing term |
| **Training Collapse After ~300 Days** | Caused by numerically unstable ODE trajectories entering training batches; Radau solver and corrected ODE eliminate the instabilities at source |
| **ODE Numerical Instabilities** | Fixed by replacing RK45 with Radau, providing the analytical Jacobian, and correcting the nitrogen mass-balance bug |
| **N Conservation Plateau** | Caused by the network learning a spurious attractor in a closed system with no nutrient supply; resolved by the open-system formulation with seasonal mixing |

---

## Version 1.0.0 — 11/03/2026

### Summary

Initial upload of all files required for the self-contained experiment, including all necessary Git configuration files.

---

### Code Structure and Default Configurations

The codebase passes all necessary information to the main file through a series of predetermined configuration files, covering data generation, training, and evaluation. Please refer to the README for a full overview of the code structure, and to the individual files for more detailed explanations.

The current default configurations are as follows (excluding constant values):

- **Data generation batch size:** 5,000 samples (random seed: 42)
- **Initial state ranges for chemical variables:**
  | Variable | Min   | Max   |
  |----------|-------|-------|
  | `N0`     | 0.50  | 10.00 |
  | `P0`     | 0.05  | 3.00  |
  | `Z0`     | 0.02  | 1.50  |
  | `D0`     | 0.00  | 2.00  |

  > Perturbation of 0.25 applied to generate distinct trajectories.
- **Model architecture:** 6 inputs, 4 outputs, 3 hidden layers (128 → 128 → 64 nodes)
- **Training batch:** 512 trajectories, max 200 epochs
- **Evaluation batch:** 100 trajectories

---

### Known Issues

#### 1. Raw Model Error

Evaluation across 10, 100, and 200 trajectories yields a consistent raw error (total mmol |ΔN|) in the range of **30–35 mmolN/m³**, regardless of the number of trajectories used. Mitigation is planned but is not the current focus of the project.

#### 2. Training Collapse After ~300 Days

All results show anomalous Neural Network behaviour around the **300-day mark**, characterised by an unexpected nutrient drop and a sharp increase in phytoplankton. This pattern is occasionally present in the ODE solutions as well (observed in roughly 5 out of 300+ trajectories), where it is likely attributable to numerical instability. These unstable ODE trajectories propagate into the training batches, contributing to the erratic NN behaviour. Fixes are planned.

#### 3. ODE Numerical Instabilities

A subset of ODE trajectories exhibit oscillating patterns consistent with **numerical instability**, producing unrealistic behaviours and N concentrations. Corrections are planned.

#### 4. Neural Network N Conservation Plateau

A distinct behaviour has been identified in the better-performing rollout trajectories. In cases where the NN and ODE trajectories agree without numerical instabilities or sudden behavioural shifts, the NN's **N conservation plateaus at ~50 mmolN/m³**, regardless of when that value is reached.

Interestingly, a dichotomy exists:
- **When the NN matches the ODE** (stable annual cycle): the N-conservation plateau is present.
- **When the NN diverges from the ODE** (e.g. the 300-day bloom): the plateau is absent and the NN's N conservation more closely tracks the ODE.