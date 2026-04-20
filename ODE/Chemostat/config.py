from dataclasses import dataclass

@dataclass
class SimParams:
    # Time Settings
    dt: float = 0.1
    total_duration: float = 150.0   
    treatment_start_time: float = 24.0 

    # Bacterial Growth
    beta: float = 2.0467      
    delta: float = 0.020467   
    K_bladder: float = 1e6    
    gamma: float = 1.0        

    # Spatial / Biofilm
    k_attach: float = 0.2     
    k_detach: float = 0.005   
    K_wall: float = 1e6       

    # Continuous Washout (Replaces void_fraction)
    k_wash: float = 0.15      

    # PD Parameters
    Psi_max: float = 0.88      
    Psi_min: float = -6.5      
    zMIC: float = 0.017        
    kappa: float = 1.1         

    # Drug Elimination
    delta_decay: float = 0.4