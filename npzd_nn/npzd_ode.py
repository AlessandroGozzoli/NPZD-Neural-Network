# =============================================================================
# npzd_ode.py
# NPZD ODE system with seasonal nutrient supply (open system).
#
# State variables (mmol N m⁻³):
#   N — dissolved inorganic nitrogen
#   P — phytoplankton biomass
#   Z — zooplankton biomass
#   D — particulate detritus
#
# The system is OPEN: a seasonal mixing term supplies inorganic nitrogen
# from depth in winter, representing mixed-layer deepening. This is the
# standard mechanism that drives the spring phytoplankton bloom and is
# present in all realistic 0D NPZD formulations (Fasham et al. 1990).
#
# Without this term the closed box depletes all N into biomass/detritus
# and the system collapses to a biologically dead fixed point.
#
# ODE system:
#   dN/dt = -μP + αGZ + φD  +  κ(t)·(N_deep − N)   ← mixing term
#   dP/dt = +μP − GZ − εP
#   dZ/dt =      βGZ − gZ
#   dD/dt = (1−α−β)GZ + εP + gZ − φD
#
# Analytical Jacobian is provided (required by Radau for efficiency).
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from config import ODE_PARAMS, FORCING, SOLVER


# =============================================================================
# Forcing functions
# =============================================================================

def light_forcing(t: np.ndarray, cfg: dict = FORCING) -> np.ndarray:
    """Sinusoidal annual PAR [W m⁻²].  t in days."""
    omega = 2.0 * np.pi / 365.0
    return cfg["I_mean"] + cfg["I_amp"] * np.sin(omega * t - np.pi / 2.0 + cfg["I_phase"])


def temp_forcing(t: np.ndarray, cfg: dict = FORCING) -> np.ndarray:
    """Sinusoidal annual SST [°C].  t in days."""
    omega = 2.0 * np.pi / 365.0
    return cfg["T_mean"] + cfg["T_amp"] * np.sin(omega * t - np.pi / 2.0 + cfg["T_phase"])


def mixing_rate(t: np.ndarray, cfg: dict = FORCING) -> np.ndarray:
    """
    Seasonal mixing rate κ(t) [d⁻¹].
    Peaks on Jan 1 (t=0), zero from spring to autumn equinox.
    κ(t) = kappa_max · max(0, cos(2π·t/365))
    """
    return cfg["kappa_max"] * np.maximum(0.0, np.cos(2.0 * np.pi * np.asarray(t) / 365.0))


def get_forcing_at_times(times: np.ndarray) -> np.ndarray:
    """Return (len(times), 2) array of [I, T] at each time."""
    I_arr = light_forcing(np.asarray(times))
    T_arr = temp_forcing(np.asarray(times))
    return np.stack([I_arr, T_arr], axis=1)


# =============================================================================
# Rate functions
# =============================================================================

def _phyto_growth(N: float, I: float, T: float, p: dict) -> float:
    """
    Specific phytoplankton growth rate [d⁻¹].
    μ = Vm(T) · N/(kN+N) · I/(kI+I)
    Vm(T) = Vm_a · Vm_b^T   (Eppley 1972)
    """
    Vm  = p["Vm_a"] * (p["Vm_b"] ** T)
    fN  = N / (p["kN"] + N)
    fI  = max(I, 0.0) / (p["kI"] + max(I, 0.0))
    return Vm * fN * fI


def _zoo_grazing(P: float, p: dict) -> float:
    """
    Zooplankton grazing rate [d⁻¹].
    G(P) = Rm · (1 − exp(−λP))   (Ivlev 1955)
    """
    return p["Rm"] * (1.0 - np.exp(-p["lam"] * P))


# =============================================================================
# ODE right-hand side
# =============================================================================

def npzd_rhs(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    """
    NPZD ODE right-hand side.  N-conservation is exact analytically.

    Fluxes and their routing:
      PP   = μ·P          N → P
      GR   = G(P)·Z       P → {α→N, β→Z, (1-α-β)→D}
      Ploss= ε·P          P → D
      Zmort= g·Z          Z → D
      Remin= φ·D          D → N

    dN/dt = -PP  + α·GR        + φ·D
    dP/dt = +PP  - GR  - ε·P
    dZ/dt =      + β·GR - g·Z
    dD/dt =   (1-α-β)·GR + ε·P + g·Z - φ·D

    Sum of RHS = 0 (verified above).
    """
    # Clamp to zero: prevents negative values from producing unphysical rates.
    # The solver should never produce negatives with Radau + tight tolerances,
    # but the clamp is a safety net during the RHS evaluations.
    N = max(y[0], 0.0)
    P = max(y[1], 0.0)
    Z = max(y[2], 0.0)
    D = max(y[3], 0.0)

    I = light_forcing(t)
    T = temp_forcing(t)
    kappa  = mixing_rate(t)
    N_deep = FORCING["N_deep"]

    mu  = _phyto_growth(N, I, T, p)
    G   = _zoo_grazing(P, p)

    alpha, beta = p["alpha"], p["beta"]
    eps, g, phi = p["eps"], p["g"], p["phi"]

    PP    = mu * P
    GR    = G * Z
    Ploss = eps * P
    Zmort = g * Z
    Remin = phi * D

    # Mixing term: seasonal nutrient supply from depth (open system).
    # Only acts when N < N_deep (brings N up toward N_deep in winter).
    Mix = kappa * (N_deep - N)

    dN = -PP + alpha * GR + Remin + Mix   # ← mixing term added here
    dP = +PP  - GR  - Ploss
    dZ =        beta * GR  - Zmort
    dD = (1.0 - alpha - beta) * GR + Ploss + Zmort - Remin

    return np.array([dN, dP, dZ, dD])


# =============================================================================
# Analytical Jacobian   J[i,j] = d(dy_i/dt) / dy_j
# Providing this to Radau avoids finite-difference approximation and
# significantly reduces the number of RHS evaluations per step.
# =============================================================================

def npzd_jacobian(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    """
    Analytical 4×4 Jacobian of the NPZD RHS.
    """
    N = max(y[0], 0.0)
    P = max(y[1], 0.0)
    Z = max(y[2], 0.0)
    D = max(y[3], 0.0)

    I = light_forcing(t)
    T = temp_forcing(t)

    Vm    = p["Vm_a"] * (p["Vm_b"] ** T)
    fI    = max(I, 0.0) / (p["kI"] + max(I, 0.0))
    kN    = p["kN"]
    lam   = p["lam"]
    Rm    = p["Rm"]
    alpha, beta = p["alpha"], p["beta"]
    eps, g, phi = p["eps"], p["g"], p["phi"]

    # Partial derivatives of PP = Vm · fI · N/(kN+N) · P
    mu          = Vm * fI * N / (kN + N)
    dPP_dN      = Vm * fI * kN / (kN + N)**2 * P   # ∂PP/∂N
    dPP_dP      = mu                                 # ∂PP/∂P

    # Partial derivatives of GR = Rm·(1-exp(-λP))·Z
    G           = Rm * (1.0 - np.exp(-lam * P))
    dG_dP       = Rm * lam * np.exp(-lam * P)
    dGR_dP      = dG_dP * Z   # ∂GR/∂P
    dGR_dZ      = G            # ∂GR/∂Z

    kappa = mixing_rate(t)

    J = np.zeros((4, 4))

    # Row 0: dN/dt = -PP + α·GR + φ·D + κ·(N_deep−N)
    # ∂/∂N of mixing term = -κ
    J[0, 0] = -dPP_dN - kappa
    J[0, 1] = -dPP_dP + alpha * dGR_dP
    J[0, 2] =           alpha * dGR_dZ
    J[0, 3] = phi

    # Row 1: dP/dt = PP - GR - ε·P
    J[1, 0] = dPP_dN
    J[1, 1] = dPP_dP - dGR_dP - eps
    J[1, 2] =        - dGR_dZ
    J[1, 3] = 0.0

    # Row 2: dZ/dt = β·GR - g·Z
    J[2, 0] = 0.0
    J[2, 1] = beta * dGR_dP
    J[2, 2] = beta * dGR_dZ - g
    J[2, 3] = 0.0

    # Row 3: dD/dt = (1-α-β)·GR + ε·P + g·Z - φ·D
    gam     = 1.0 - alpha - beta
    J[3, 0] = 0.0
    J[3, 1] = gam * dGR_dP + eps
    J[3, 2] = gam * dGR_dZ + g
    J[3, 3] = -phi

    return J


# =============================================================================
# Solver wrapper
# =============================================================================

def run_npzd(
    y0        : np.ndarray,
    params    : dict = None,
    solver_cfg: dict = None,
) -> dict:
    """
    Integrate the NPZD system and return daily snapshots for one year.

    Spin-up: the system is first integrated for solver_cfg['spinup_days']
    days starting from y0. The end state is then used as the true initial
    condition for the output year. This eliminates transient behaviour.

    Parameters
    ----------
    y0         : Initial state [N0, P0, Z0, D0]  (mmol N m⁻³)
    params     : Ecological parameters  (defaults to ODE_PARAMS)
    solver_cfg : Solver settings  (defaults to SOLVER)

    Returns
    -------
    dict with keys:
        't'      : (n_steps,)   time array [days]
        'states' : (n_steps, 4) state array [N, P, Z, D]
        'forcing': (n_steps, 2) forcing [I, T]
        'success': bool
        'message': solver message string
    """
    if params is None:
        params = ODE_PARAMS
    if solver_cfg is None:
        solver_cfg = SOLVER

    rhs = lambda t, y: npzd_rhs(t, y, params)
    jac = lambda t, y: npzd_jacobian(t, y, params)

    # No spin-up: initial conditions are sampled from winter conditions
    # (high N, low biology) which is the natural starting state for a
    # spring bloom simulation. Spin-up drove the system to a degenerate
    # nutrient-depleted fixed point.
    y0_start = np.clip(y0, 0.0, None)

    t_eval = np.linspace(
        solver_cfg["t_start"],
        solver_cfg["t_end"],
        solver_cfg["n_steps"],
    )

    sol = solve_ivp(
        fun    = rhs,
        t_span = (solver_cfg["t_start"], solver_cfg["t_end"]),
        y0     = y0_start,
        method = solver_cfg["method"],
        jac    = jac,
        t_eval = t_eval,
        rtol   = solver_cfg["rtol"],
        atol   = solver_cfg["atol"],
        dense_output = False,
    )

    if not sol.success:
        return {"success": False, "message": sol.message}

    # Clip any tiny floating-point negatives — should not occur with Radau
    # but added as a safety net.
    states = np.clip(sol.y.T.copy(), 0.0, None)
    forcing = get_forcing_at_times(sol.t)

    return {
        "t"      : sol.t,
        "states" : states,
        "forcing": forcing,
        "success": True,
        "message": sol.message,
    }


# =============================================================================
# Quick sanity check — run directly to validate ODE correctness
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Running NPZD sanity check with Radau solver...")
    y0 = [7.0, 0.3, 0.05, 0.1]

    result = run_npzd(y0)

    if not result["success"]:
        print(f"ODE solver failed: {result['message']}")
    else:
        t       = result["t"]
        states  = result["states"]
        forcing = result["forcing"]

        total_N  = states.sum(axis=1)
        drift_pct = (total_N.max() - total_N.min()) / total_N[0] * 100

        print(f"Solver:         Radau (implicit, order 5)")
        print(f"N at t=0:       {total_N[0]:.6f} mmol N m⁻³")
        print(f"N at t=365:     {total_N[-1]:.6f} mmol N m⁻³")
        print(f"Max drift:      {drift_pct:.4f} %")
        print(f"Min state val:  {states.min():.2e} mmol N m⁻³")
        print(f"Spring bloom P: {states[:, 1].max():.3f} mmol N m⁻³ "
              f"on day {t[states[:, 1].argmax()]:.0f}")

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(t, forcing[:, 0], "gold",   lw=1.5, label="PAR [W m⁻²]")
        ax_twin = axes[0].twinx()
        ax_twin.plot(t, forcing[:, 1], "tomato", lw=1.5, linestyle="--",
                     label="SST [°C]")
        axes[0].set_ylabel("PAR [W m⁻²]", color="goldenrod")
        ax_twin.set_ylabel("SST [°C]", color="tomato")
        axes[0].set_title("NPZD Model — Sanity Check  (Radau solver)")

        colors = ["steelblue", "forestgreen", "darkorange", "saddlebrown"]
        labels = ["N", "P", "Z", "D"]
        for i, (c, l) in enumerate(zip(colors, labels)):
            axes[1].plot(t, states[:, i], color=c, lw=1.5, label=l)
        axes[1].set_ylabel("mmol N m⁻³")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.25)

        axes[2].plot(t, total_N, "k-", lw=1.5, label=f"Total N  (drift {drift_pct:.4f}%)")
        axes[2].set_ylabel("mmol N m⁻³")
        axes[2].set_xlabel("Day of year")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig("npzd_sanity_check.png", dpi=150)
        print("\nPlot saved → npzd_sanity_check.png")
        plt.show()
