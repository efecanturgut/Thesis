import numpy as np

def birthrate(N, beta, delta, K, gamma):
    birth_rate = beta - gamma*(beta-delta) * N/K
    if np.isscalar(birth_rate):
        if birth_rate < 0:
            birth_rate = 0
    else:
        birth_rate[birth_rate < 0] = 0
    return birth_rate

def natural_deathrate(N, beta, delta, K, gamma):
    natural_death = delta + (1-gamma)*(beta-delta) * N/K
    return natural_death

def antibiotic_death_rate(A, Psi_max, Psi_min, zMIC, kappa):
    A = np.maximum(1e-12, A)
    ratio_term_power = (A / zMIC)**kappa
    numerator = (Psi_max - Psi_min) * ratio_term_power
    denominator = ratio_term_power - (Psi_min / Psi_max)
    mu_log10 = np.divide(numerator, denominator)
    return mu_log10 * np.log(10)

def step_antibiotic_exponential(A_curr, dt, alpha):
    return A_curr * np.exp(-alpha * dt)