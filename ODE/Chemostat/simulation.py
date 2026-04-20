import numpy as np
from scipy.integrate import solve_ivp
from config import SimParams
from functions import birthrate, natural_deathrate, antibiotic_death_rate

def bladder_odes(t, y, p: SimParams):
    P, Att, A = y
    
    P = max(0.0, P)
    Att = max(0.0, Att)
    A = max(0.0, A)

    # Rates
    mu_P_nat = natural_deathrate(P, p.beta, p.delta, p.K_bladder, p.gamma)
    mu_P_drug = antibiotic_death_rate(A, p.Psi_max, p.Psi_min, p.zMIC, p.kappa)
    b_P = birthrate(P, p.beta, p.delta, p.K_bladder, p.gamma)

    mu_Att_nat = natural_deathrate(Att, p.beta, p.delta, p.K_wall, p.gamma)
    mu_Att_drug = antibiotic_death_rate(A, p.Psi_max, p.Psi_min, p.zMIC, p.kappa)
    b_Att = birthrate(Att, p.beta, p.delta, p.K_wall, p.gamma)

    # Derivatives
    dP = (b_P - mu_P_nat - mu_P_drug) * P - (p.k_attach * max(0.0, 1.0 - Att / p.K_wall) * P) + (p.k_detach * Att) - (p.k_wash * P)
    dAtt = (b_Att - mu_Att_nat - mu_Att_drug) * Att + (p.k_attach * max(0.0, 1.0 - Att / p.K_wall) * P) - (p.k_detach * Att)
    dA = -p.delta_decay * A

    return [dP, dAtt, dA]

def run_ode_sweep(p: SimParams, T_dose, dose_concentration, return_history=False):
    y0 = [1e4, 0.0, 0.0]  # [P_curr, Att_curr, A_curr]
    
    # Pre-calculate integration segments based on dosing intervals
    dose_times = np.arange(p.treatment_start_time, p.total_duration + T_dose, T_dose)
    time_points = np.unique(np.concatenate(([0.0, p.total_duration], dose_times)))
    time_points = time_points[time_points <= p.total_duration]

    if return_history:
        t_hist = [0.0]
        P_hist = [y0[0]]
        Att_hist = [y0[1]]
        A_hist = [y0[2]]

    for i in range(len(time_points) - 1):
        t_start = time_points[i]
        t_end = time_points[i+1]

        # Apply dose if integration segment aligns with a dose time
        if np.any(np.isclose(t_start, dose_times)):
            y0[2] += dose_concentration

        # Extinction threshold (Asymptotic correction)
        if y0[0] + y0[1] < 1.0:
            y0[0] = 0.0
            y0[1] = 0.0
            if not return_history:
                return False

        sol = solve_ivp(
            fun=bladder_odes,
            t_span=[t_start, t_end],
            y0=y0,
            args=(p,),
            method='LSODA',
            max_step=p.dt
        )

        # Update initial conditions for next segment
        y0 = sol.y[:, -1]
        y0[0] = max(0.0, y0[0])
        y0[1] = max(0.0, y0[1])
        y0[2] = max(0.0, y0[2])

        if return_history:
            t_hist.extend(sol.t[1:])
            P_hist.extend(sol.y[0, 1:])
            Att_hist.extend(sol.y[1, 1:])
            A_hist.extend(sol.y[2, 1:])

    if return_history:
        return t_hist, P_hist, Att_hist, A_hist

    return (y0[0] + y0[1]) > 1.0