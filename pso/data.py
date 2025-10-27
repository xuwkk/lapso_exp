import os
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['Weekday_sin', 
                'Weekday_cos',
                'Hour_sin',
                'Hour_cos',
                'Temperature (k)',
                'Shortwave Radiation (w/m2)',
                'Longwave Radiation (w/m2)',
                'Zonal Wind Speed (m/s)',
                'Meridional Wind Speed (m/s)',
                'Wind Speed (m/s)']

def get_dataset_np(grid_cfg):
    """Load and preprocess power system data from CSV files.
    
    Args:
        grid_xlsx: Dictionary containing grid data from Excel file
        data_path: Path to directory containing bus data CSV files
        
    Returns:
        feature_all: Array of features for each bus [data_no, no_bus, no_features]
        load_all: Array of load values scaled by baseMVA [data_no, no_bus]
        solar_all: Array of solar values scaled by baseMVA [data_no, no_solar] or None
        wind_all: Array of wind values scaled by baseMVA [data_no, no_wind] or None
    """
    # Initialize empty lists to store data
    print('Returning the dataset...')
    load_all, solar_all, wind_all = [], [], []
    feature_all = []
    
    # Get indices of buses with non-zero load (1-based indexing)
    baseMVA = grid_cfg['baseMVA']
    
    file_list = os.listdir(grid_cfg['data_dir'])
    file_list = [file for file in file_list if file.startswith('bus_') and file.endswith('.csv')]
    # Sort files by bus index
    bus_indices = [int(f.split('_')[1].split('.')[0]) for f in file_list]
    file_list = [x for _, x in sorted(zip(bus_indices, file_list))]
    # print('The file list is: ', file_list)
    
    # Load data for each bus
    for i in range(len(file_list)):
        data = pd.read_csv(os.path.join(grid_cfg['data_dir'], file_list[i]))
        
        # Extract features and load data
        feature_all.append(data.iloc[:,:len(FEATURE_COLUMNS)].values)  # [data_no, no_features]
        load_all.append(data['Load'].values)
        
        # Only append solar/wind if present at this bus
        if np.sum(data['Solar']) > 0:
            solar_all.append(data['Solar'].values)
        if np.sum(data['Wind']) > 0:
            wind_all.append(data['Wind'].values)

    # Reshape and scale data
    # Convert [no_bus, data_no, features] to [data_no, no_bus, features]
    feature_all = np.stack(feature_all, axis=0).transpose(1, 0, 2)
    
    # Scale load by base MVA
    load_all = np.array(load_all).T / baseMVA
    
    # Process solar and wind if present
    solar_all = None if len(solar_all) == 0 else np.array(solar_all).T / baseMVA
    wind_all = None if len(wind_all) == 0 else np.array(wind_all).T / baseMVA
    
    # # Rescale the solar by shifting the entire solar curve
    # renewable = 0
    # if solar_all is not None:
    #     renewable += np.sum(solar_all, axis=1) # aggregate all solar per hour
    # if wind_all is not None:
    #     renewable += np.sum(wind_all, axis=1)
    
    # renewable_load_ratio_max_ = np.max(renewable / load_all.sum(axis=1)) # maximum ratio per hour
    
    # if grid_cfg['renewable_load_ratio_max'] is not None:
    #     if solar_all is not None:
    #         solar_all = solar_all / renewable_load_ratio_max_ * grid_cfg['renewable_load_ratio_max']
    #     if wind_all is not None:
    #         wind_all = wind_all / renewable_load_ratio_max_ * grid_cfg['renewable_load_ratio_max']
    
    if solar_all is not None:   
        print('solar/load max per hour: ', np.max(np.sum(solar_all, axis=1) / np.sum(load_all, axis=1)))

    # if grid_cfg['renewable_min'] > 0.0:
    #     if solar_all is not None:
    #         solar_all = np.clip(solar_all, grid_cfg['renewable_min']/baseMVA, None)
    #     if wind_all is not None:
    #         wind_all = np.clip(wind_all, grid_cfg['renewable_min']/baseMVA, None)
    
    return feature_all, load_all, solar_all, wind_all