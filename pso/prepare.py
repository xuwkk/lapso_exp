import numpy as np
import pandas as pd
import pypower.api as pp
import os
import shutil
from pso.solve_opt import solve_opt

def prepare_grid_from_pypower(grid_cfg:dict, force_new:bool=False, random_seed = 0):
    """
    Combine the pypower case with the extra config (in .yaml) and save it as an excel file
    Return: a excel file, each sheet is a dataframe for a specific sheet such as bus, gen, branch, solar, wind, etc.
    ! The IDX entries in the excel file are starting from 1
    """

    np.random.seed(random_seed)
    
    if not force_new:
        if not os.path.exists(grid_cfg['config_xlsx_path']):
            print(f"Grid config file {grid_cfg['config_xlsx_path']} does not exist. Please set the force_new = True to create the grid config.")
        else:
            print(f"Grid config file {grid_cfg['config_xlsx_path']} exists. Please set the force_new = True to create the grid config.")
            grid_xlsx = pd.read_excel(grid_cfg['config_xlsx_path'], sheet_name=None, engine='openpyxl')
            return grid_xlsx
    
    sheet_columns = {
        # the definition must be the same as MATPOWER
        "bus": ["BUS_I", "BUS_TYPE", "PD", "QD", "GS", "BS", "BUS_AREA", "VM", "VA", "BASEKV", "ZONE", "VMAX", "VMIN"],
        "gen": ["GEN_BUS", "PG", "QG", "QMAX", "QMIN", "VG", "MBASE", "GEN_STATUS", "PMAX", "PMIN"],
        "branch": ["F_BUS", "T_BUS", "BR_R", "BR_X", "BR_B", "RATE_A", "RATE_B", "RATE_C", "TAP", "SHIFT", "BR_STATUS", "ANGMIN", "ANGMAX"],
        "gencost": ["MODEL", "STARTUP", "SHUTDOWN", "ORDER", "SECOND", "FIRST", "ZERO"]
        }
    
    
    # Obtain the pypower entries
    ppc = getattr(pp, grid_cfg['pypower_case_name'])()
    sheet_dict = {}
    for key, value in ppc.items():
        if key not in ['version', 'baseMVA']:
            sheet_dict[key] = pd.DataFrame(value[:, :len(sheet_columns[key])], 
                                columns=sheet_columns[key])

    """
    Reset the index to start from 1 and end at no_bus
    """
    default_bus_idx = sheet_dict['bus']['BUS_I'].values
    target_bus_id = np.arange(1, len(default_bus_idx) + 1)
    bus_idx_map = {default_bus_idx[i]: target_bus_id[i] for i in range(len(default_bus_idx))}

    # reset buses
    sheet_dict['bus']['BUS_I'] = target_bus_id
    # reset generators
    sheet_dict['gen']['GEN_BUS'] = sheet_dict['gen']['GEN_BUS'].map(bus_idx_map)
    # reset branches
    sheet_dict['branch']['F_BUS'] = sheet_dict['branch']['F_BUS'].map(bus_idx_map)
    sheet_dict['branch']['T_BUS'] = sheet_dict['branch']['T_BUS'].map(bus_idx_map)
    
    """
    Merge the existing pypower config with the extra config
    """
    random_config_ratio = grid_cfg['random_ratio'] if 'random_ratio' in grid_cfg else 0.0  # when the format is ratio, allow certain randomness
    
    for key, value in grid_cfg['extra_config'].items():
        if key in sheet_dict:
            # existing entries of the pypower case, for example, bus, gen, branch, gencost
            for col_name, config in value.items():
                # each item of entries
                if config['format'] == 'value':
                    # set the value directly
                    if len(config['value']) == 1:
                        # if only one value is provided, apply it to all buses
                        sheet_dict[key][col_name] = config['value'] * np.ones(sheet_dict[key].shape[0])
                    elif len(config['value']) == sheet_dict[key].shape[0]:
                        sheet_dict[key][col_name] = config['value']
                    else:
                        raise ValueError(f"The length of the value {config['value']} does not match the number of rows in the {key} sheet")
                
                elif config['format'] == "ratio":
                    # defined by the ratio of an generator PMAX or FIRST order cost
                    if key == "gen":
                        # generator limits are subject to the pmax
                        pmax = sheet_dict[key]['PMAX'].values
                        if len(config['value']) == 1:
                            # if only one value is provided, apply it to all buses subject to a random ratio
                            ratio = config['value'][0] * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, len(pmax)))
                        elif len(config['value']) == sheet_dict[key].shape[0]:
                            ratio = np.array(config['value'])
                            ratio = ratio * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, len(pmax)))
                        else:
                            raise ValueError(f"The length of the value {config['value']} does not match the number of rows in the {key} sheet")

                        sheet_dict[key][col_name] = ratio * pmax

                    elif key == "gencost":
                        # generator cost are subject to the first order cost
                        first = sheet_dict[key]['FIRST'].values
                        if len(config['value']) == 1:
                            # if only one value is provided, apply it to all buses subject to a random ratio
                            ratio = config['value'][0] * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, len(first)))
                        elif len(config['value']) == sheet_dict[key].shape[0]:
                            ratio = np.array(config['value'])
                            ratio = ratio * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, len(first)))
                        else:
                            raise ValueError(f"The length of the value {config['value']} does not match the number of rows in the {key} sheet")

                        sheet_dict[key][col_name] = ratio * first

                    elif key == "bus":
                        # load shedding cost is with respect to the first order cost of the maximum generator cost
                        first = np.max(sheet_dict['gencost']['FIRST'].values)
                        if len(config['value']) == 1:
                            # if only one value is provided, apply it to all buses subject to a random ratio
                            ratio = config['value'][0] * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, sheet_dict[key].shape[0]))
                        elif len(config['value']) == sheet_dict[key].shape[0]:
                            ratio = np.array(config['value'])
                            ratio = ratio * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, sheet_dict[key].shape[0]))
                        else:
                            raise ValueError(f"The length of the value {config['value']} does not match the number of rows in the {key} sheet")

                        sheet_dict[key][col_name] = ratio * first

                    else:
                        raise ValueError(f"Unexpected key {key}.")

                else:
                    raise ValueError(f"The format {config['format']} is not supported")
                
        else:
            # Construct a new sheet from the dictionary, for example, solar and wind
            # which is not originally as part of pypower entries
            # ! currently allow solar and wind locate on the same bus
            # todo: check if this will cause issue in operation
            sheet_dict[key] = pd.DataFrame(columns=value.keys())
            for col_name, config in value.items():
                if config['format'] == 'value':
                    sheet_dict[key][col_name] = config['value']
                elif config['format'] == 'ratio':
                    # if key in ['solar', 'wind']:
                    if col_name == 'INDEX':
                        no_bus = sheet_dict['bus'].shape[0]
                        # randomly choose the bus index for placing renewable
                        # ! must be a load and non-gen bus
                        load_bus_idx = np.where(sheet_dict['bus']['PD'] > 0)[0] + 1         # start from 1
                        generator_bus_idx = sheet_dict['gen']['GEN_BUS'].values.astype(int) # start from 1
                        no_renewable = int(config['value'][0] * no_bus)
                        load_bus_idx_ = [i for i in load_bus_idx if i not in generator_bus_idx]
                        idx = np.random.choice(load_bus_idx_, no_renewable, replace=False)
                        # print(idx)
                        # print(len(idx))
                        sheet_dict[key][col_name] = idx
                    elif col_name == 'CAPACITY_RATIO':
                        # randomly choose the capacity ratio for each renewable that 
                        # sum up to the total penetration level
                        agg_ratio = config['value'][0]
                        ratios = np.random.rand(no_renewable)
                        ratios = ratios / np.sum(ratios) * agg_ratio
                        sheet_dict[key][col_name] = ratios
                    elif col_name == 'CURTAIL':
                        # with respect to the first-order cost of the maximum generator cost
                        first_max = np.max(sheet_dict['gencost']['FIRST'].values)
                        if len(config['value']) == 1:
                            # if only one value is provided, apply it to all buses
                            ratio = config['value'][0] * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, no_renewable))
                            sheet_dict[key][col_name] = ratio * first_max * np.ones(no_renewable)
                        elif len(config['value']) == no_renewable:
                            ratio = np.array(config['value'])
                            ratio = ratio * (1 + np.random.uniform(-random_config_ratio, random_config_ratio, no_renewable))
                            sheet_dict[key][col_name] = ratio * first_max

                else:
                    raise ValueError(f"The format {config['format']} is not supported")
    

    """
    Rescale the load and renewable capacity
    """

    gen_cap = np.sum(sheet_dict['gen']['PMAX']) # maximum generation capacity per hour
    if grid_cfg['rescale_load']:
        # Rescale the maximum load so that when a max generator is offline, 
        # the remaining capacity can just meet the load
        target_load_cap = gen_cap - np.max(sheet_dict['gen']['PMAX'])
        ratio = target_load_cap / np.sum(sheet_dict['bus']['PD'])
        sheet_dict['bus']['PD'] = sheet_dict['bus']['PD'] * ratio
    
    load_cap = np.sum(sheet_dict['bus']['PD'])
    # Rescale the renewable so that the maximum renewable penetration equals to the specified ratio aggregated load level
    if 'solar' in sheet_dict.keys():
        sheet_dict['solar']['CAPACITY'] = sheet_dict['solar']['CAPACITY_RATIO'] * load_cap
    if 'wind' in sheet_dict.keys():
        sheet_dict['wind']['CAPACITY'] = sheet_dict['wind']['CAPACITY_RATIO'] * load_cap

    # Convert the sheet_dict into excel sheets
    with pd.ExcelWriter(grid_cfg['config_xlsx_path']) as writer:
        for key, value in sheet_dict.items():
            value.to_excel(writer, sheet_name=key, index=False)
    
    print(f">>>>>> Grid prepared successfully and saved to {grid_cfg['config_xlsx_path']}")
    
    return pd.read_excel(grid_cfg['config_xlsx_path'], sheet_name=None, engine='openpyxl')

def prepare_data(grid_xlsx, grid_cfg, random_seed:int, force_new:bool=False):
    """
    Assign and rescale the data from the collected buses to a specific grid config
    If the number of loads and renewables are larger than the available data, 
        it will randomly reuse the assigned data
    """
    
    np.random.seed(random_seed)
    
    data_dir = grid_cfg['data_dir']

    if not force_new:
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist. Please set the force_new = True to create the data.")
        else:
            print(f"Data directory {data_dir} exists. Please set the force_new = True to create the data.")
            return
    
    if os.path.exists(data_dir):
        print(f"The data directory {data_dir} exists. It is deleted to generate new data.")
        shutil.rmtree(data_dir)
    
    bus_config = grid_xlsx['bus']
    load_bus_idx = np.where(bus_config['PD'] > 0)[0] + 1  # start from 1
    no_load = len(load_bus_idx)
    no_solar = 0 if 'solar' not in grid_xlsx.keys() else len(grid_xlsx['solar'])
    no_wind = 0 if 'wind' not in grid_xlsx.keys() else len(grid_xlsx['wind'])

    # A dictionary contain all the data for each load bus (start from 1)
    # !Not all buses have load data
    load_data_all = {key: [] for key in load_bus_idx}  

    data_grouped_dir = 'data/bus_data'
    file_name = os.listdir(data_grouped_dir)

    def assign_solar_or_wind_data(load_data_all, renew_name, assigned_name, no_renew):
        """helper function to assign solar or wind buses"""
        
        assigned_renew_no = 0
        if renew_name in grid_xlsx.keys():
            renew_config = grid_xlsx[renew_name]
            for i in range(len(renew_config)):
                renew_bus_idx = renew_config['INDEX'][i]
                
                for name in file_name:
                    # Find a bus data file containing solar/wind data that hasn't been used yet
                    if not name.endswith('.csv') or name in assigned_name:
                        continue
                        
                    data = pd.read_csv(os.path.join(data_grouped_dir, name))
                    if np.sum(data[renew_name.capitalize()]) <= 0 or np.sum(data['Load']) <= 0:
                        # only consider solar/wind located at the load bus
                        continue
                        
                    assigned_name.append(name)
                    
                    # Rescale load and solar/wind data
                    default_load = bus_config['PD'][renew_bus_idx - 1]
                    data['Load'] = data['Load'] * default_load / np.max(data['Load'])
                    
                    default_renew = renew_config['CAPACITY'][i] 
                    data[renew_name.capitalize()] = data[renew_name.capitalize()] * default_renew / np.max(data[renew_name.capitalize()])
                    if renew_name == 'solar':
                        data['Wind'] = 0.0
                    else:
                        data['Solar'] = 0.0
                    
                    load_data_all[renew_bus_idx] = data
                    
                    assigned_renew_no += 1
                    break

            # after going through all the data files, check if enough solar/wind data has been assigned
            if assigned_renew_no < no_renew:
                print(f"Warning: only {assigned_renew_no} {renew_name} data have been assigned, less than the required {no_renew}")
                print(f"Repeated {renew_name} data is randomly assigned to the remaining {renew_name} buses and rescale")

                for i in range(assigned_renew_no, no_renew):
                    # index of this renew bus
                    renew_bus_idx = renew_config['INDEX'][i]
                    # pick one from the load_data_all that has solar/wind
                    # by design, it already has load
                    available_idx = [key for key in load_data_all.keys() if len(load_data_all[key]) > 0 and np.sum(load_data_all[key][renew_name.capitalize()]) > 0]
                    chosen_idx = np.random.choice(available_idx, 1)[0]
                    data = load_data_all[chosen_idx].copy()
                    # Rescale load and solar/wind data
                    default_load = bus_config['PD'][renew_bus_idx - 1]
                    data['Load'] = data['Load'] * default_load / np.max(data['Load'])
                    default_renew = renew_config['CAPACITY'][i]
                    data[renew_name.capitalize()] = data[renew_name.capitalize()] * default_renew / np.max(data[renew_name.capitalize()])
                    load_data_all[renew_bus_idx] = data
    
        return load_data_all, assigned_name

    assigned_name = [] # trace the file names that have been assigned
    # assign solar and wind
    load_data_all, assigned_name = assign_solar_or_wind_data(load_data_all, 'solar', assigned_name, no_solar)
    load_data_all, assigned_name = assign_solar_or_wind_data(load_data_all, 'wind', assigned_name, no_wind)


    ## for the remaining load: randomly choose
    remaining_file_name = [name for name in file_name if name not in assigned_name and '.csv' in name]
    assigned_load_no = len([key for key in load_data_all.keys() if len(load_data_all[key]) > 0])
    required_load_no = no_load - assigned_load_no
    if len(remaining_file_name) >= required_load_no:
        
        # the remaining data is sufficient: randomly choose the remaining data files
        remaining_file_name = np.random.choice(remaining_file_name, required_load_no, replace=False)
        idx = 0
        for i in load_bus_idx:
            if len(load_data_all[i]) == 0: # the load has not been assigned
                name = remaining_file_name[idx]
                data = pd.read_csv(os.path.join(data_grouped_dir, name))
                # Rescale load
                max_load = np.max(data['Load'])
                default_load = bus_config['PD'][i - 1]
                data['Load'] = data['Load'] * default_load / max_load
                data['Solar'] = 0.0     # pure load bus
                data['Wind'] = 0.0
                load_data_all[i] = data
                assigned_name.append(name)
                idx += 1
    else:
        # the remaining data is not sufficient
        print(f"Warning: only {len(remaining_file_name)} remaining data files, less than the required {required_load_no}")
        print(f"Warning: not enough remaining data files, randomly choose one from the assigned data")
        idx = 0
        for i in load_bus_idx:
            if len(load_data_all[i]) == 0: # the load has not been assigned
                try:
                    # assign the remaining data files first
                    name = remaining_file_name[idx]
                    data = pd.read_csv(os.path.join(data_grouped_dir, name))
                    # Rescale load
                    max_load = np.max(data['Load'])
                    default_load = bus_config['PD'][i - 1]
                    data['Load'] = data['Load'] * default_load / max_load
                    data['Solar'] = 0.0     # pure load bus
                    data['Wind'] = 0.0
                    load_data_all[i] = data
                    assigned_name.append(name)
                    idx += 1
                except:
                    # if not enough remaining files, randomly choose one from the assigned data
                    available_idx = [key for key in load_data_all.keys() if len(load_data_all[key]) > 0]
                    chosen_idx = np.random.choice(available_idx, 1)[0]
                    data = load_data_all[chosen_idx].copy()
                    # Rescale load
                    default_load = bus_config['PD'][i - 1]
                    data['Load'] = data['Load'] * default_load / np.max(data['Load'])
                    data['Solar'] = 0.0
                    data['Wind'] = 0.0
                    load_data_all[i] = data
                    idx += 1

    
    """
    clip/rescale the renewable min and max
    min is given by the config,
    max is given by the renewable_load_ratio_max such that the maximum per hour penetration is maintained
    """
    load_all, renewable_all = 0, 0
    for i in load_bus_idx:
        data = load_data_all[i]
        load_all += data['Load'].values # aggregated per hour
        renewable_all += data['Solar'].values + data['Wind'].values # aggregated per hour
    
    renewable_load_ratio_max_ = np.max(renewable_all / load_all) # maximum ratio per hour
    # print('Before rescaling, solar+wind/load max per hour: ', renewable_load_ratio_max_)
    
    if grid_cfg['renewable_load_ratio_max'] is not None:
        global_ratio = grid_cfg['renewable_load_ratio_max'] / renewable_load_ratio_max_
        for i in load_bus_idx:
            data = load_data_all[i].copy()
            if np.sum(data['Solar']) > 0:
                data['Solar'] = data['Solar'] * global_ratio
                data['Solar'] = np.clip(data['Solar'], grid_cfg['renewable_min'], None)
            if np.sum(data['Wind']) > 0:
                data['Wind'] = data['Wind'] * global_ratio
                data['Wind'] = np.clip(data['Wind'], grid_cfg['renewable_min'], None)
            load_data_all[i] = data

    # save the data
    os.makedirs(data_dir, exist_ok=True)
    for i in load_bus_idx:
        load_data_all[i].to_csv(os.path.join(data_dir, f'bus_{i}.csv'), index=False)
    
    print(f">>>>>> Data prepared successfully and saved to {data_dir}")

def refine_config(load, solar, wind, grid_cfg:dict, optimization_cfg:dict, uc):
    """
    Automatically modify the power flow limit based on the initial UC results
    grid_cfg: the grid config
    optimization_cfg: the optimization config
    uc: the uc model (continuous or discrete)
    """
    print("========== Modifying the power flow limit... ==========")
    
    if not grid_cfg['rescale_line_limit']['force_new']:
        print(f"force_new_data is set to False, please set it to True to modify the maximum branch limits in place")
        return
    
    min_pfmax = grid_cfg['rescale_line_limit']['min_pfmax']
    scale_factor = grid_cfg['rescale_line_limit']['scale_factor']
    
    # Start at the maximum generator output
    uc.pfmax = np.ones(uc.no_branch) * np.max(uc.pmax)
    # print(f"The initial pfmax is {uc.pfmax}")
    uc.formulate() # Formulate the uc problem
    print('uc.with_pf_constraint: ', uc.with_pf_constraint)
    
    uc.optimization_summary()
    uc.system_summary()
    
    # Prepare data
    no_sample = load.shape[0] // uc.T

    load_batch = np.array([load[i * uc.T:(i + 1) * uc.T] for i in range(no_sample)])
    solar_batch = np.array([solar[i * uc.T:(i + 1) * uc.T] for i in range(no_sample)]) if solar is not None else None
    wind_batch = np.array([wind[i * uc.T:(i + 1) * uc.T] for i in range(no_sample)]) if wind is not None else None

    print("No of samples: ", len(solar_batch))
    print(f"Data sizes: load {load_batch.shape}, solar {None if solar_batch is None else solar_batch.shape}, wind {None if wind_batch is None else wind_batch.shape}")
    
    # Solve the uc problem
    uc_results = solve_opt(uc, load_batch, solar_batch, wind_batch, optimization_cfg)
    
    """analysis"""
    # print(f"uc_results: {uc_results[0]}")
    ug_summary = np.concatenate([result["ug"] for result in uc_results], axis=0)
    print(f"Each generator on ratio: {np.sum(ug_summary, axis=0) / len(ug_summary)}")
    
    ls_summary = np.concatenate([result["ls"] for result in uc_results], axis=0)
    ls_rate = len(np.where(ls_summary.flatten() > 1e-6)[0]) / len(ls_summary.flatten())
    print(f'ls rate (ave. per hour): {ls_rate}')
    
    if solar is not None:
        solarc_summary = np.concatenate([result["solarc"] for result in uc_results], axis=0)
        solarc_rate = len(np.where(solarc_summary.flatten() > 1e-6)[0]) / len(solarc_summary.flatten())
        print(f'solarc rate (ave. per hour): {solarc_rate}')
    if wind is not None:
        windc_summary = np.concatenate([result["windc"] for result in uc_results], axis=0)
        windc_rate = len(np.where(windc_summary.flatten() > 1e-6)[0]) / len(windc_summary.flatten())
        print(f'windc rate (ave. per hour): {windc_rate}')
    
    """save the rescaled pfmax"""
    pf_summary = np.concatenate([result["power_flow"] for result in uc_results], axis=0)
    pf_max = np.max(np.abs(pf_summary), axis=0)
    print('Ave. Max PF from optimization:', np.mean(pf_max))
    
    # Randomly change the pfmax from -scale_factor to +scale_factor
    pf_max_rescaled = pf_max * (1 + np.random.uniform(-scale_factor, scale_factor, len(pf_max)))
    # Set a minimum value of pfmax
    pf_max_rescaled = np.clip(pf_max_rescaled * uc.baseMVA, a_min=min_pfmax, a_max=None)
    print('Ave. Max pf rescaled:', np.mean(pf_max_rescaled))
    
    # modify the maximum branch limits from the xlsx file in place
    config = pd.read_excel(grid_cfg['config_xlsx_path'], sheet_name=None, engine='openpyxl')
    config['branch']['RATE_A'] = pf_max_rescaled
    
    with pd.ExcelWriter(grid_cfg['config_xlsx_path']) as writer:
        for key, value in config.items():
            value.to_excel(writer, sheet_name=key, index=False)
    
    print(f"The maximum branch limits have been rescaled and saved to {grid_cfg['config_xlsx_path']}")





    # assigned_solar_no = 0
    # if 'solar' in grid_xlsx.keys():
    #     solar_config = grid_xlsx['solar']
    #     for i in range(len(solar_config)):
    #         solar_bus_idx = solar_config['INDEX'][i]
            
    #         for name in file_name:
    #             if not name.endswith('.csv') or name in assigned_name:
    #                 # Find a bus data file containing solar data that hasn't been used yet
    #                 continue
                    
    #             data = pd.read_csv(os.path.join(data_grouped_dir, name))
    #             if np.sum(data['Solar']) <= 0 or np.sum(data['Load']) <= 0:
    #                 # only consider solar located at the load bus
    #                 continue
                    
    #             assigned_name.append(name)
                
    #             # Rescale load and solar data
    #             default_load = bus_config['PD'][solar_bus_idx - 1]
    #             data['Load'] = data['Load'] * default_load / np.max(data['Load'])
                
    #             default_solar = solar_config['CAPACITY'][i] 
    #             data['Solar'] = data['Solar'] * default_solar / np.max(data['Solar'])
                
    #             load_data_all[solar_bus_idx] = data
    #             assigned_solar_no += 1
    #             break


    #     # after going through all the data files, check if enough solar data has been assigned
    #     if assigned_solar_no < no_solar:
    #         print(f"Warning: only {assigned_solar_no} solar data have been assigned, less than the required {no_solar}")
    #         print("Repeated solar data is randomly assigned to the remaining solar buses and rescale")

    #         for i in range(assigned_solar_no, no_solar):
    #             # index of this solar bus
    #             solar_bus_idx = solar_config['INDEX'][i]
    #             # pick one from the load_data_all that has solar
    #             # by design, it already has load
    #             available_idx = [key for key in load_data_all.keys() if len(load_data_all[key]) > 0 and np.sum(load_data_all[key]['Solar']) > 0]
    #             chosen_idx = np.random.choice(available_idx, 1)[0]
    #             data = load_data_all[chosen_idx].copy()
    #             # Rescale load and solar data
    #             default_load = bus_config['PD'][solar_bus_idx - 1]
    #             data['Load'] = data['Load'] * default_load / np.max(data['Load'])
    #             default_solar = solar_config['CAPACITY'][i]
    #             data['Solar'] = data['Solar'] * default_solar / np.max(data['Solar'])
    #             load_data_all[solar_bus_idx] = data
            
    # assigned_wind_no = 0
    # if 'wind' in grid_xlsx.keys():
    #     wind_config = grid_xlsx['wind']
    #     for i in range(len(wind_config)):
    #         wind_bus_idx = wind_config['INDEX'][i]
            
    #         # Find a bus data file containing wind data that hasn't been used yet
    #         for name in file_name:
    #             if not name.endswith('.csv') or name in assigned_name:
    #                 # each bus either have solar or wind
    #                 continue
                    
    #             data = pd.read_csv(os.path.join(data_grouped_dir, name))
    #             if np.sum(data['Wind']) <= 0 or np.sum(data['Load']) <= 0:
    #                 continue
                    
    #             assigned_name.append(name)
                
    #             # Rescale load and wind data
    #             default_load = bus_config['PD'][wind_bus_idx - 1]
    #             data['Load'] = data['Load'] * default_load / np.max(data['Load'])
                
    #             default_wind = wind_config['CAPACITY'][i]
    #             data['Wind'] = data['Wind'] * default_wind / np.max(data['Wind'])
                
    #             load_data_all[wind_bus_idx] = data
    #             assigned_wind_no += 1
    #             break
        
        # if assigned_wind_no < no_wind:
        #     print(f"Warning: only {assigned_wind_no} wind data have been assigned, less than the required {no_wind}")
        #     print("Repeated wind data is randomly assigned to the remaining wind buses")

        #     for i in range(assigned_wind_no, no_wind):
        #         # index of this wind bus
        #         wind_bus_idx = wind_config['INDEX'][i]
        #         # pick one from the load_data_all that has wind
        #         available_idx = [key for key in load_data_all.keys() if len(load_data_all[key]) > 0 and np.sum(load_data_all[key]['Wind']) > 0]
        #         chosen_idx = np.random.choice(available_idx, 1)[0]
        #         data = load_data_all[chosen_idx].copy()
        #         # Rescale load and wind data
        #         default_load = bus_config['PD'][wind_bus_idx - 1]
        #         data['Load'] = data['Load'] * default_load / np.max(data['Load'])
        #         default_wind = wind_config['CAPACITY'][i]
        #         data['Wind'] = data['Wind'] * default_wind / np.max(data['Wind'])
        #         load_data_all[wind_bus_idx] = data


    # if len(remaining_file_name) > 0:
    #     # there are remaining data files
    #     remaining_file_name = np.random.choice(remaining_file_name, no_load - len(assigned_name), replace=False)
    #     idx = 0
    #     for i in load_bus_idx:
    #         if len(load_data_all[i]) == 0: # the load has not been assigned
    #             name = remaining_file_name[idx]
    #             data = pd.read_csv(os.path.join(data_grouped_dir, name))
    #             # Rescale load
    #             max_load = np.max(data['Load'])
    #             default_load = bus_config['PD'][i - 1]
    #             data['Load'] = data['Load'] * default_load / max_load
    #             data['Solar'] = 0     # pure load bus
    #             data['Wind'] = 0
    #             load_data_all[i] = data
    #             assigned_name.append(name)
    #             idx += 1
    # else:
    #     # there is no remaining data files
    #     idx = 0
    #     for i in load_bus_idx:
    #         if len(load_data_all[i]) == 0:
    #             # if not enough remaining files, randomly choose one from the assigned data
    #             print(f"Warning: not enough remaining data files, randomly choose one from the assigned data")
    #             available_idx = [key for key in load_data_all.keys() if len(load_data_all[key]) > 0]
    #             chosen_idx = np.random.choice(available_idx, 1)[0]
    #             data = load_data_all[chosen_idx].copy()
    #             # Rescale load
    #             default_load = bus_config['PD'][i - 1]
    #             data['Load'] = data['Load'] * default_load / np.max(data['Load'])
    #             data['Solar'] = 0
    #             data['Wind'] = 0
    #             load_data_all[i] = data
    #             idx += 1


