# =============================================================================
# npzd_ode.py
# Implements the NPZD (Nutrient-Phytoplankton-Zooplankton-Detritus)
# biogeochemical ODE system and its scipy-based solver.
#
# State variables (all in mmol N m^-3):
#   N  — dissolved inorganic nitrogen (nutrient)
#   P  — phytoplankton nitrogen biomass
#   Z  — zooplankton nitrogen biomass
#   D  — particulate detritus nitrogen
#
# Environmental forcing (time-varying):
#   I(t) — photosynthetically active radiation [W m^-2]
#   T(t) — sea surface temperature [°C]
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from config import ODE_PARAMS, FORCING, SOLVER


# =============================================================================
# Forcing functions
# =============================================================================

def light_forcing(t: float, cfg: dict = FORCING) -> float:
    """
    Sinusoidal annual PAR cycle.
    Maximum occurs around day 91 (spring equinox) when phase = 0.
    t in days (0 = Jan 1).
    """
    omega = 2.0 * np.pi / 365.0
    return cfg["I_mean"] + cfg["I_amp"] * np.sin(omega * t - np.pi / 2.0 + cfg["I_phase"])


def temp_forcing(t: float, cfg: dict = FORCING) -> float:
    """
    Sinusoidal annual SST cycle.
    Lags light by ~T_phase radians (temperature response delayed vs. insolation).
    """
    omega = 2.0 * np.pi / 365.0
    return cfg["T_mean"] + cfg["T_amp"] * np.sin(omega * t - np.pi / 2.0 + cfg["T_phase"])


def get_forcing_at_times(times: np.ndarray) -> np.ndarray:
    """
    Evaluate both forcing variables at an array of times.
    Returns array of shape (len(times), 2) with columns [I, T].
    """
    I = np.array([light_forcing(t) for t in times])
    T = np.array([temp_forcing(t) for t in times])
    return np.stack([I, T], axis=1)


# =============================================================================
# Growth and grazing rate functions
# =============================================================================

def phyto_growth_rate(N: float, I: float, T: float, p: dict) -> float:
    """
    Nutrient- and light-limited phytoplankton specific growth rate.

    Uses:
      - Eppley-type temperature dependence:  Vm(T) = Vm_a * Vm_b^T
      - Michaelis-Menten for nutrient:        f_N   = N / (kN + N)
      - Michaelis-Menten for light:           f_I   = I / (kI + I)
      - Overall: mu = Vm(T) * f_N * f_I

    Returns mu [d^-1].
    """
    Vm = p["Vm_a"] * (p["Vm_b"] ** T)
    f_N = max(N, 0.0) / (p["kN"] + max(N, 0.0))
    f_I = max(I, 0.0) / (p["kI"] + max(I, 0.0))
    return Vm * f_N * f_I


def zoo_grazing_rate(P: float, p: dict) -> float:
    """
    Ivlev (1955) saturating grazing functional response.
    G(P) = Rm * (1 - exp(-lambda * P))   [d^-1]
    """
    return p["Rm"] * (1.0 - np.exp(-p["lam"] * max(P, 0.0)))


# =============================================================================
# The NPZD ODE right-hand side
# =============================================================================

def npzd_rhs(t: float, y: np.ndarray, params: dict) -> list:
    """
    Right-hand side of the NPZD ODE system.

    Parameters
    ----------
    t      : current time [days]
    y      : state vector [N, P, Z, D]  (mmol N m^-3)
    params : dict of ecological parameters (from config.ODE_PARAMS or perturbed copy)

    Returns
    -------
    dydt : list of derivatives [dN/dt, dP/dt, dZ/dt, dD/dt]
    """
    N, P, Z, D = y

    # Clamp to zero to prevent numerical blow-up of negatives
    N = max(N, 0.0)
    P = max(P, 0.0)
    Z = max(Z, 0.0)
    D = max(D, 0.0)

    # Environmental forcing at time t
    I = light_forcing(t)
    T = temp_forcing(t)

    # Derived rates
    mu  = phyto_growth_rate(N, I, T, params)   # Phyto growth rate [d^-1]
    G   = zoo_grazing_rate(P, params)           # Zoo grazing rate  [d^-1]

    alpha = params["alpha"]
    beta  = params["beta"]
    eps   = params["eps"]
    g     = params["g"]
    phi   = params["phi"]

    # --- Flux terms ---
    # Phytoplankton primary production
    PP   = mu * P                       # [mmol N m^-3 d^-1]

    # Zooplankton grazing on phytoplankton
    GR   = G * Z                        # [mmol N m^-3 d^-1]

    # Partitioning of grazing:
    #   alpha fraction -> dissolved N  (excretion)
    #   beta  fraction -> zooplankton biomass
    #   (1-alpha-beta) fraction -> detritus  (sloppy feeding / unassimilated)
    G_to_N = alpha * GR
    G_to_Z = beta  * GR
    G_to_D = (1.0 - alpha - beta) * GR

    # Phytoplankton loss (excretion + mortality) -> nutrient pool
    P_loss = eps * P

    # Zooplankton mortality -> detritus
    Z_mort = g * Z

    # Remineralization of detritus -> nutrient
    Remin  = phi * D

    # --- ODEs ---
    dN_dt = -PP + G_to_N + P_loss + Remin
    dP_dt =  PP - GR     - P_loss
    dZ_dt =  G_to_Z      - Z_mort
    dD_dt =  G_to_D      + P_loss + Z_mort - Remin

    return [dN_dt, dP_dt, dZ_dt, dD_dt]


# =============================================================================
# Solver wrapper
# =============================================================================

def run_npzd(
    y0     : np.ndarray,
    params : dict = None,
    solver_cfg: dict = None,
) -> dict:
    """
    Solve the NPZD system from t=0 to t=365 days and return daily snapshots.

    Parameters
    ----------
    y0         : Initial state [N0, P0, Z0, D0]
    params     : Ecological parameter dict (defaults to config.ODE_PARAMS)
    solver_cfg : Solver settings dict (defaults to config.SOLVER)

    Returns
    -------
    result : dict with keys:
        't'       — time array (n_steps,)
        'states'  — state array (n_steps, 4)  [N, P, Z, D]
        'forcing' — forcing array (n_steps, 2) [I, T]
        'success' — bool
    """
    if params is None:
        params = ODE_PARAMS
    if solver_cfg is None:
        solver_cfg = SOLVER

    t_eval = np.linspace(
        solver_cfg["t_start"],
        solver_cfg["t_end"],
        solver_cfg["n_steps"]
    )

    sol = solve_ivp(
        fun     = lambda t, y: npzd_rhs(t, y, params),
        t_span  = (solver_cfg["t_start"], solver_cfg["t_end"]),
        y0      = y0,
        method  = solver_cfg["method"],
        t_eval  = t_eval,
        rtol    = solver_cfg["rtol"],
        atol    = solver_cfg["atol"],
    )

    forcing = get_forcing_at_times(sol.t)

    return {
        "t"       : sol.t,
        "states"  : sol.y.T,        # shape (n_steps, 4)
        "forcing" : forcing,         # shape (n_steps, 2)
        "success" : sol.success,
    }


# =============================================================================
# Quick sanity check (run this file directly to see a plot)
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    y0 = [8.0, 0.2, 0.1, 0.1]   # Typical winter initial conditions

    result = run_npzd(y0)

    if not result["success"]:
        print("ODE solver failed!")
    else:
        t      = result["t"]
        states = result["states"]
        forcing = result["forcing"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(t, forcing[:, 0], color="gold", label="PAR [W/m²]")
        axes[0].plot(t, forcing[:, 1] * 5, color="tomato", linestyle="--",
                     label="Temperature [°C] × 5")
        axes[0].set_ylabel("Forcing")
        axes[0].legend(fontsize=8)
        axes[0].set_title("NPZD Model — Single Trajectory (Sanity Check)")

        colors = ["steelblue", "green", "darkorange", "saddlebrown"]
        labels = ["N (Nutrient)", "P (Phytoplankton)", "Z (Zooplankton)", "D (Detritus)"]
        for i, (c, l) in enumerate(zip(colors, labels)):
            axes[1].plot(t, states[:, i], color=c, label=l)
        axes[1].set_ylabel("mmol N m⁻³")
        axes[1].legend(fontsize=8)

        total_N = states.sum(axis=1)
        axes[2].plot(t, total_N, color="black", label="Total N (conservation check)")
        axes[2].set_ylabel("mmol N m⁻³")
        axes[2].set_xlabel("Day of Year")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig("npzd_sanity_check.png", dpi=150)
        print("Plot saved to npzd_sanity_check.png")
        plt.show()

        print(f"\nN conservation: min={total_N.min():.4f}, max={total_N.max():.4f}, "
              f"drift={total_N.max()-total_N.min():.4f} mmol N m^-3")