import numpy as np
import pandas as pd
import pypower.api as pp
import os
from tqdm import trange
import datetime

def return_renewable_incidences():
    """Return the solar and wind idx to the bus idx,
    each bus can only have one solar or wind"""

    # link the generator idx to the bus idx
    # one bus can have multiple gen
    gen_data = pd.read_excel('data/Data_public/Generator_data.xlsx', sheet_name='Gen data')    
    # link the solar idx to the gen idx
    solar_data = pd.read_excel('data/Data_public/Generator_data.xlsx', sheet_name='Solar Plant Number')
    # link the wind idx to the gen idx
    wind_data = pd.read_excel('data/Data_public/Generator_data.xlsx', sheet_name='Wind Plant Number')

    GenIdx_to_BusIdx = {}
    for i in range(len(gen_data)):
        GenIdx_to_BusIdx[gen_data["Gen Number"][i]] = gen_data["Bus Number"][i]

    SolarIdx_to_GenIdx = {}
    for i in range(len(solar_data)):
        SolarIdx_to_GenIdx[solar_data["Solar Plant Number"][i]] = solar_data["Generator Number"][i]

    WindIdx_to_GenIdx = {}
    for i in range(len(wind_data)):
        WindIdx_to_GenIdx[wind_data["Wind Plant Number"][i]] = wind_data["Generator Number"][i]
    
    # Link the solar and wind idx to the bus idx
    SolarIdx_to_BusIdx = {}
    for solar_idx, gen_idx in SolarIdx_to_GenIdx.items():
        SolarIdx_to_BusIdx[solar_idx] = GenIdx_to_BusIdx[gen_idx]

    WindIdx_to_BusIdx = {}
    for wind_idx, gen_idx in WindIdx_to_GenIdx.items():
        WindIdx_to_BusIdx[wind_idx] = GenIdx_to_BusIdx[gen_idx]
    
    # Remove the solar and wind idx if
    registered_bus_idx = []
    SolarIdx_to_BusIdx_new, WindIdx_to_BusIdx_new = {}, {}
    for solar_idx, bus_idx in SolarIdx_to_BusIdx.items():
        if bus_idx not in registered_bus_idx:
            registered_bus_idx.append(bus_idx)
            SolarIdx_to_BusIdx_new[solar_idx] = bus_idx
    for wind_idx, bus_idx in WindIdx_to_BusIdx.items():
        if bus_idx not in registered_bus_idx:
            registered_bus_idx.append(bus_idx)
            WindIdx_to_BusIdx_new[wind_idx] = bus_idx
    
    return SolarIdx_to_BusIdx_new, WindIdx_to_BusIdx_new

def preprocess_data():
    save_dir = './data/bus_data'
    
    os.makedirs(save_dir, exist_ok=True)
    
    no_bus, no_day, no_hour = 123, 365, 24
    
    # empty dataframes
    climate_first_hour = pd.read_excel("data/Data_public/Climate_2019/climate_2019_Day" + '1.csv', sheet_name='Hour 1')
    data_all = {key: pd.DataFrame(columns=climate_first_hour.columns) for key in range(1, no_bus+1)}
    
    # Climate data: a dictionary, each key is a bus, which contains per hour: 24*365 rows
    for day in trange(1, no_day+1, desc='Loading climate data'):
        # day 1 to 365
        climate_file = f"data/Data_public/Climate_2019/climate_2019_Day{day}.csv"
        climate_data_day = pd.ExcelFile(climate_file)
        
        for hour in range(1, no_hour+1):
            # hour 1 to 24
            climate_data_hour = climate_data_day.parse(f'Hour {hour}')
            
            for bus_idx in range(1, no_bus+1):
                row_data = climate_data_hour.iloc[bus_idx-1:bus_idx]
                if len(data_all[bus_idx]) == 0:
                    data_all[bus_idx] = row_data
                else:
                    data_all[bus_idx] = pd.concat([data_all[bus_idx], row_data], 
                                                ignore_index=True, axis=0)

    # Load data
    load_all = []
    for day in trange(1, no_day+1, desc='Loading load data'):
        load_all.append(pd.read_csv(f'data/Data_public/load_2019/load_annual_D{day}.txt', sep=" ", header=None))
    load_all = pd.concat(load_all, axis=0)  # (no_day*no_hour, no_bus)
    load_all.reset_index(drop=True, inplace=True)
    for bus_idx in range(1, no_bus+1):
        data_all[bus_idx]['Load'] = load_all[bus_idx-1] # Add the load column to the existing dataframe of climate data
    
    # Solar and wind incidence
    solar_idx_to_bus_idx, wind_idx_to_bus_idx = return_renewable_incidences()
    
    # Add the solar and wind data
    for bus_idx in range(1, no_bus+1):
        data_all[bus_idx]['Solar'] = 0
        data_all[bus_idx]['Wind'] = 0
    
    # All the solar data
    solar_all = {}
    for solar_idx in solar_idx_to_bus_idx.keys():
        solar_data = []
        for day in range(1, no_day+1):
            # for each day, extract the solar data of the specific solar idx
            solar_day = pd.read_csv(f'data/Data_public/solar_2019/solar_annual_D{day}.txt', sep=" ", header=None)
            solar_data.append(solar_day.iloc[solar_idx-1,:])
        solar_all[solar_idx] = pd.concat(solar_data, axis=0, ignore_index=True)
    
    for solar_idx, bus_idx in solar_idx_to_bus_idx.items():
        data_all[bus_idx]['Solar'] = solar_all[solar_idx]  # add column solar to the existing dataframe
    
    # All the wind data
    wind_all = {}
    for wind_idx in wind_idx_to_bus_idx.keys():
        wind_data = []
        for day in range(1, no_day+1):
            wind_day = pd.read_csv(f'data/Data_public/wind_2019/wind_annual_D{day}.txt', sep=" ", header=None)
            wind_data.append(wind_day.iloc[wind_idx-1,:])
        wind_all[wind_idx] = pd.concat(wind_data, axis=0, ignore_index=True)
    
    for wind_idx, bus_idx in wind_idx_to_bus_idx.items():
        data_all[bus_idx]['Wind'] = wind_all[wind_idx]
    
    # Add calender data
    start_weekday = datetime.datetime(2019,1,1).weekday()
    one_week = np.concatenate([np.arange(start_weekday, 7), (np.arange(0, start_weekday))])

    day = np.repeat(np.arange(1,no_day + 1), 24)
    hour = np.tile(np.arange(1,25), no_day)
    weekday = np.tile(np.repeat(one_week, 24), 53)[:no_day * 24]

    # use sin and consine to capture the periodicity
    hour_sin = np.sin(2 * np.pi * ( hour / 24))
    hour_cos = np.cos(2 * np.pi * ( hour / 24))
    weekday_sin = np.sin(2 * np.pi * ( weekday / 7))
    weekday_cos = np.cos(2 * np.pi * ( weekday / 7))
    
    for bus in range(1, no_bus+1):
        data_all[bus]['Hour_sin'] = hour_sin
        data_all[bus]['Hour_cos'] = hour_cos
        data_all[bus]['Weekday_sin'] = weekday_sin
        data_all[bus]['Weekday_cos'] = weekday_cos
        data_all[bus]['Load'] = load_all.iloc[:,bus-1]
    
    # change the order of the columns
    columns = ['Weekday_sin', 'Weekday_cos', 'Hour_sin', 'Hour_cos', 'Temperature (k)', 'Shortwave Radiation (w/m2)',
                    'Longwave Radiation (w/m2)', 'Zonal Wind Speed (m/s)',
                    'Meridional Wind Speed (m/s)', 'Wind Speed (m/s)',
                    'Load', 'Solar', 'Wind']
    
    for bus, data in data_all.items():
        data[columns].to_csv(f'{save_dir}/bus_{bus}.csv', index=False)

if __name__ == "__main__":
    preprocess_data()