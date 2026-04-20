import numpy as np
from config import SimParams
from functions import (
    birthrate, natural_deathrate, antibiotic_death_rate, step_antibiotic_exponential
)

def run_single_sweep(p: SimParams, T_dose, dose_concentration, rng: np.random.Generator, return_history=False):
    P_curr = 1e4   
    Att_curr = 0.0 
    I_cells_curr = np.zeros(p.N_cells) 
    A_curr = 0.0 
    
    t_curr = 0.0
    next_dose_time = p.treatment_start_time
    total_elimination_rate = p.delta_decay # + p.k_wash

    if return_history:
        t_hist = [t_curr]; P_hist = [P_curr]; Att_hist = [Att_curr]
        I_hist = [np.sum(I_cells_curr)]; A_hist = [A_curr]
    
    while t_curr < p.total_duration:
        
        # --- 0. OPTIMIZATION: Early Exit ---
        if t_curr > p.treatment_start_time:
            total_bacteria = P_curr + Att_curr + np.sum(I_cells_curr)
            if total_bacteria == 0:
                if return_history:
                    while t_curr < p.total_duration:
                        if t_curr >= next_dose_time:
                            A_curr = dose_concentration
                            next_dose_time += T_dose
                            
                        A_curr = step_antibiotic_exponential(A_curr, p.dt, total_elimination_rate)
                        t_curr += p.dt
                        
                        t_hist.append(t_curr)
                        P_hist.append(0); Att_hist.append(0); I_hist.append(0); A_hist.append(A_curr)
                    return t_hist, P_hist, Att_hist, I_hist, A_hist
                else:
                    return False

        # --- 1. Dosing Event ---
        if t_curr >= p.treatment_start_time:
            if t_curr >= next_dose_time:
                A_curr = dose_concentration 
                next_dose_time += T_dose
        
        drug_conc_bladder = A_curr
        drug_conc_cell = A_curr * p.permeability
        
        mu_P_nat = natural_deathrate(P_curr, p.beta, p.delta, p.K_bladder, p.gamma)
        mu_P_drug = antibiotic_death_rate(drug_conc_bladder, p.Psi_max, p.Psi_min, p.zMIC, p.kappa)
        b_P = birthrate(P_curr, p.beta, p.delta, p.K_bladder, p.gamma)
        
        mu_Att_nat = natural_deathrate(Att_curr, p.beta, p.delta, p.K_wall, p.gamma)
        mu_Att_drug = antibiotic_death_rate(drug_conc_bladder, p.Psi_max, p.Psi_min, p.zMIC, p.kappa)
        b_Att = birthrate(Att_curr, p.beta, p.delta, p.K_wall, p.gamma)
        
        b_cells = birthrate(I_cells_curr, p.beta_cell, p.delta_cell, p.K_single_cell, p.gamma)
        mu_cells_nat = natural_deathrate(I_cells_curr, p.beta_cell, p.delta_cell, p.K_single_cell, p.gamma)
        mu_cells_drug = antibiotic_death_rate(drug_conc_cell, p.Psi_cell_max, p.Psi_cell_min, p.zMIC, p.kappa)
        
        # --- Planktonic Compartment (Sequential Updates) ---
        births_P = rng.poisson(b_P * P_curr * p.dt)
        deaths_P_nat  = min(rng.poisson(mu_P_nat  * P_curr * p.dt), int(P_curr))
        deaths_P_drug = min(rng.poisson(mu_P_drug * P_curr * p.dt), max(0, int(P_curr) - deaths_P_nat))
        
        available_P_for_attach = max(0, int(P_curr) - deaths_P_nat - deaths_P_drug)
        prob_attach = min(1.0, max(0.0, p.k_attach * (1.0 - Att_curr/p.K_wall) * p.dt))
        flow_attach = rng.binomial(available_P_for_attach, prob_attach)
        
        available_P_for_washout = max(0, available_P_for_attach - flow_attach)
        prob_washout = min(1.0, max(0.0, p.k_wash * p.dt))
        flow_washout = rng.binomial(available_P_for_washout, prob_washout)

        # --- Attached Compartment (Sequential Updates) ---
        births_Att = rng.poisson(b_Att * Att_curr * p.dt)
        deaths_Att_nat  = min(rng.poisson(mu_Att_nat  * Att_curr * p.dt), int(Att_curr))
        deaths_Att_drug = min(rng.poisson(mu_Att_drug * Att_curr * p.dt), max(0, int(Att_curr) - deaths_Att_nat))

        available_Att_for_detach = max(0, int(Att_curr) - deaths_Att_nat - deaths_Att_drug)
        prob_detach = min(1.0, max(0.0, p.k_detach * p.dt))
        flow_detach = rng.binomial(available_Att_for_detach, prob_detach)

        available_Att_for_invade = max(0, available_Att_for_detach - flow_detach)
        prob_invade = min(1.0, max(0.0, p.k_invade * p.dt))
        total_invaders = rng.binomial(available_Att_for_invade, prob_invade)

        # --- Intracellular Compartment ---
        births_I = rng.poisson(b_cells * I_cells_curr * p.dt)
        deaths_I_nat  = np.minimum(rng.poisson(mu_cells_nat  * I_cells_curr * p.dt), I_cells_curr.astype(int))
        deaths_I_drug = np.minimum(rng.poisson(mu_cells_drug * I_cells_curr * p.dt), np.maximum(0, I_cells_curr.astype(int) - deaths_I_nat))
        
        I_cells_next = np.maximum(0, I_cells_curr + births_I - deaths_I_nat - deaths_I_drug)
        
        if total_invaders > 0:
            target_indices = rng.integers(0, p.N_cells, size=total_invaders)
            np.add.at(I_cells_next, target_indices, 1)

        burst_mask = I_cells_next > p.burst_threshold
        bacteria_released = np.sum(I_cells_next[burst_mask])
        I_cells_next[burst_mask] = 0 
        
        # --- Aggregation ---
        P_next = P_curr + births_P - deaths_P_nat - deaths_P_drug - flow_attach + flow_detach + bacteria_released - flow_washout
        Att_next = Att_curr + births_Att - deaths_Att_nat - deaths_Att_drug + flow_attach - flow_detach - total_invaders
        
        A_next = step_antibiotic_exponential(A_curr, p.dt, total_elimination_rate)
        
        P_curr = max(0, P_next)
        Att_curr = max(0, Att_next)
        I_cells_curr = I_cells_next
        A_curr = A_next
        t_curr += p.dt
        
        if return_history:
            t_hist.append(t_curr)
            P_hist.append(P_curr)
            Att_hist.append(Att_curr)
            I_hist.append(np.sum(I_cells_curr))
            A_hist.append(A_curr)

    if return_history:
        return t_hist, P_hist, Att_hist, I_hist, A_hist
    
    final_total = P_curr + Att_curr + np.sum(I_cells_curr)
    return final_total > 0