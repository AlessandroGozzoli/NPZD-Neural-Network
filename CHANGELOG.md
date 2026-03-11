# Changelog

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