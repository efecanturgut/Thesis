import numpy as np

def birthrate(N, beta, delta, K, gamma):
    birth_rate = beta - gamma*(beta-delta) * N/K
    return max(0.0, birth_rate)

def natural_deathrate(N, beta, delta, K, gamma):
    return delta + (1-gamma)*(beta-delta) * N/K

def antibiotic_death_rate(A, Psi_max, Psi_min, zMIC, kappa):
    A = max(1e-12, A)
    ratio_term_power = (A / zMIC)**kappa
    numerator = (Psi_max - Psi_min) * ratio_term_power
    denominator = ratio_term_power - (Psi_min / Psi_max)
    return (numerator / denominator) * np.log(10)