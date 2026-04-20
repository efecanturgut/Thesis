# %%

import time
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

from config import SimParams
from simulation import run_single_sweep

# ==============================================================================
# 0. PARALLEL WORKER FUNCTION
# ==============================================================================
def evaluate_pixel(args):
    """
    Executes the full Monte Carlo ensemble for a single coordinate.
    """
    # Unpack the concentration index (c_idx) alongside the spatial coordinates
    c_idx, i, j, t_void, t_dose, dose_conc, p, num_iters = args
    
    ensemble_survivals = np.zeros(num_iters, dtype=bool)
    
    for k in range(num_iters):
        ensemble_survivals[k] = run_single_sweep(
            p=p,
            T_void=t_void, 
            T_dose=t_dose, 
            dose_concentration=dose_conc,
            return_history=False
        )
    
    probability_of_failure = np.mean(ensemble_survivals)
    # Return the c_idx so the main process knows which heatmap to update
    return c_idx, i, j, probability_of_failure

# ==============================================================================
# MAIN EXECUTION BLOCK (M1 SPATIAL PROTECTION)
# ==============================================================================
if __name__ == '__main__':
    params = SimParams()
    params.print_settings()

    Conc_Multipliers = [0.5, 1.0, 1.5, 8.0] 
    Conc_Labels = [f"Dose = {m}x MIC" for m in Conc_Multipliers]

    grid_steps = 47
    T_void_vec = np.linspace(1.0, 24.0, grid_steps)
    T_dose_vec = np.linspace(1.0, 24.0, grid_steps)

    num_iterations = 10  
    
    # Expand matrix to handle 4 doses: (4, Y, X)
    Results_Matrix = np.zeros((4, grid_steps, grid_steps))

    print(f"\nStarting Full Parallel Sweep ({num_iterations} iterations per pixel)...")
    start_time = time.time()

    # Task Assembly
    tasks = []
    for c_idx, mult in enumerate(Conc_Multipliers):
        current_dose_conc = mult * params.zMIC
        for i, t_dose in enumerate(T_dose_vec):
            for j, t_void in enumerate(T_void_vec):
                # Pack c_idx into the tuple
                task_args = (c_idx, i, j, t_void, t_dose, current_dose_conc, params, num_iterations)
                tasks.append(task_args)

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(evaluate_pixel, tasks)

    # Matrix Reassembly
    for c_idx, i, j, prob in results:
        Results_Matrix[c_idx, i, j] = prob

    elapsed = time.time() - start_time
    print(f"Parallel simulation completed in {elapsed:.2f} seconds.")

    # ==============================================================================
    # 5. PLOTTING (2x2 Grid)
    # ==============================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    axs = axs.ravel()

    extent = [T_void_vec[0], T_void_vec[-1], T_dose_vec[0], T_dose_vec[-1]]

    for idx in range(4):
        ax = axs[idx]
        data = Results_Matrix[idx, :, :]
        
        im = ax.imshow(data, origin='lower', extent=extent, aspect='auto',
                        cmap='RdYlBu_r', vmin=0.0, vmax=1.0)
        
        ax.set_title(Conc_Labels[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel("Voiding Interval (h)", fontsize=10)
        ax.set_ylabel("Treatment Interval (h)", fontsize=10)
        
        # Phase boundary contour at 50% probability
        ax.contour(data, levels=[0.5], extent=extent, colors='white', linestyles='--')

    # Global Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Establishment Probability (Failure Rate)", fontsize=14)

    plt.suptitle(f"Probability of Treatment Failure (N={num_iterations} replicates)", fontsize=18, fontweight='bold')
    plt.show()
# %%
