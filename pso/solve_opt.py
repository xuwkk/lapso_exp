import numpy as np
from functools import partial
from tqdm import tqdm
def solve_opt(uc, load_forecast, solar_forecast, wind_forecast, optimization_cfg:dict,
              rd = None, load_true = None, solar_true = None, wind_true = None):
    """
    Solve the optimization problem
    """
    T = uc.T
    if load_forecast is not None: # (no_sample, T, no_feature)
        assert load_forecast.ndim == 3, "The load should be at least 3D array"
        assert load_forecast.shape[1] == T, "The number of load data should be equal to the number of hours"
    if solar_forecast is not None:
        assert solar_forecast.ndim == 3, "The solar should be at least 3D array"
        assert solar_forecast.shape[1] == T, "The number of solar data should be equal to the number of hours"
    if wind_forecast is not None:
        assert wind_forecast.ndim == 3, "The wind should be at least 3D array"
        assert wind_forecast.shape[1] == T, "The number of wind data should be equal to the number of hours"
    
    solver = optimization_cfg['solver']
    solver_options = optimization_cfg['option']
    
    no_sample = load_forecast.shape[0]
    
    # Partial functions for the optimization
    solve_uc = partial(uc.solve, solver=solver, **solver_options)
    uc_results = []
    
    if rd is not None:
        solve_rd = partial(rd.solve, solver=solver, **solver_options)
        rd_results = []
    
    for i in tqdm(range(no_sample)):
        uc_parameters = {
            'load': load_forecast[i], 
            'solar': solar_forecast[i] if solar_forecast is not None else None, 
            'wind': wind_forecast[i] if wind_forecast is not None else None
        }
        
        uc_result = solve_uc(parameters=uc_parameters)
        # uc_result = uc.solve(parameters=uc_parameters)
        uc_results.append(uc_result)
        if rd is not None:
            rd_parameters = {
                'load': load_true[i], 
                'solar': solar_true[i] if solar_true is not None else None, 
                'wind': wind_true[i] if wind_true is not None else None,
                'pg_parameter': uc_result['pg'], 
                'ug_parameter': uc_result['ug'] if 'ug' in uc_result.keys() else None
            }
            rd_result = solve_rd(rd_parameters)
            rd_results.append(rd_result)
    
    return (uc_results, rd_results) if rd is not None else uc_results