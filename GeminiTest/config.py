from dataclasses import dataclass

@dataclass
class SimParams:
    # Time Settings
    dt: float = 0.1
    total_duration: float = 150.0   
    sampling_window: float = 24.0   
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

    # Intracellular
    k_invade: float = 0.001         
    N_cells: int = 250             
    K_single_cell: float = 4000      
    burst_threshold: float = 3600    
    
    # Dependent variables
    beta_cell: float = None        
    delta_cell: float = None       

    # PD Parameters
    Psi_max: float = 0.88      
    Psi_min: float = -6.5      
    Psi_cell_max: float = 0.44 
    Psi_cell_min: float = -3.25 
    zMIC: float = 0.017        
    kappa: float = 1.1         
    permeability: float = 0.90 

    # Enzyme / Voiding
    production_factor: float = 0.0     
    kappa_deg: float = 0.0                 
    delta_decay: float = 0.4           
    void_fraction: float = 0.9 

    def __post_init__(self):
        if self.beta_cell is None:
            self.beta_cell = self.beta / 2
        if self.delta_cell is None:
            self.delta_cell = self.beta_cell / 100

    def print_settings(self):
        print("=== SIMULATION SETTINGS ===")
        print(f"Time: dt={self.dt}, duration={self.total_duration}, start={self.treatment_start_time}, window={self.sampling_window}")
        print(f"Bacterial: beta={self.beta}, delta={self.delta}, K_bladder={self.K_bladder}, gamma={self.gamma}")
        print(f"Biofilm: k_attach={self.k_attach}, k_detach={self.k_detach}, K_wall={self.K_wall}")
        print(f"Intracellular: k_invade={self.k_invade}, N_cells={self.N_cells}, burst={self.burst_threshold}")
        print(f"PD Params: zMIC={self.zMIC}, kappa={self.kappa}, permeability={self.permeability}")
        print(f"Enzyme/Voiding: void_fraction={self.void_fraction}, production={self.production_factor}")
        print("===========================")