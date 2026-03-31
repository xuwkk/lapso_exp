"""
Objective-based forecast with multiple uncertainty sources: worst-case analysis only.
- ML-uncertainty (adversarial attack like unceratiny)
- Opt-Uncertainty (exogenous uncertainty in optimization)
"""

import sys
sys.path.append('./')
import hydra
from omegaconf import DictConfig
from pso.data import get_dataset_np
from pso.operation import UC_CONTINUOUS, RD
import numpy as np
import os
from pso.prepare import prepare_grid_from_pypower
from paper_exp.obf_func import (data_preprocess, 
                                train_abf_model, 
                                train_obj_kkt_model_reduced,
                                evaluate_forecast,
                                evaluate_opt,
                                worst_uncertainty,
                                random_uncertainty,
                                ccg,
                                worst_uncertainty_multi
                                )
from functools import partial
import torch
from paper_exp.train_abf_nn import MLP
from tqdm import tqdm
import time

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    # to have LS work properly, the number of data should be large than the number of features
    NO_DATA = cfg.exp.train_config.no_data
    M_DP = cfg.exp.train_config.M_DP
    M_RD = cfg.exp.train_config.M_RD
    ALPHA = cfg.exp.train_config.alpha
    VERBOSE = cfg.exp.train_config.verbose
    SOLAR_MIN_CLIP = cfg.grid.renewable_min / cfg.grid.baseMVA
    BUDGET_RATIO = cfg.exp.budget_ratio                         # size of the uncertainty set
    BUDGET_RATIO_INPUT = cfg.exp.budget_ratio_input              # size of the uncertainty set for the input data
    BUDGET_RATIO_FORECAST = cfg.exp.budget_ratio_forecast        # size of the uncertainty set for the forecast
    SAVING_DIR = cfg.exp.saving_dir
    
    grid_name = cfg.grid.data_dir.split('/')[-2]
    NN_DIR = cfg.exp.train_config.nn_dir + grid_name + '/acc_forecaster.pth'
    SAVING_DIR = SAVING_DIR + f"{grid_name}/"                   # e.g., "paper_exp/obf_result/obf_uncer/bus14/"
    os.makedirs(SAVING_DIR, exist_ok=True)
    
    assert cfg.operation.with_binary == False, "Only consider continuous case for now"
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    
    # Load data
    feature_total, load_total, solar_total, _ = get_dataset_np(cfg.grid)
    NO_DATA_TOTAL = feature_total.shape[0]
    SOLAR_MAX = np.max(solar_total, axis = 0)
    
    # Load optimization models
    uc = UC_CONTINUOUS(grid_xlsx, cfg.operation)  # The horizon is 1-hour in DP
    uc.formulate()
    rd = RD(grid_xlsx, cfg.operation)
    rd.formulate(discrete_uc=False)
    
    # Test if generation + renewable capacity is enough to cover the load
    print('ug_init: ', uc.ug_init)
    pg_max = (uc.pmax * uc.ug_init).sum()
    gen_diff = pg_max + np.sum(solar_total, axis = 1) - load_total.sum(axis = 1)
    print('gen_diff min: ', np.min(gen_diff), 'gen_diff max: ', np.max(gen_diff))
    assert np.all(gen_diff >= 0), "Generation + renewable capacity is not enough to cover the load"
    
    # Preprocess the whole dataset: mainly normalize the feature
    feature_total, load_total, solar_total = data_preprocess(feature_total, load_total, solar_total)

    # Load NN
    model = MLP(feature_total.shape[1], solar_total.shape[1])
    model.load_state_dict(torch.load(NN_DIR))
    model.to('cpu').eval()
    nn_forecast, nn_extracted = model(torch.from_numpy(feature_total).float())
    nn_extracted = nn_extracted.detach().numpy()
    
    # Train the OBF model on each week (168 hours)
    # Obtain the sample_idx
    sample_idx = NO_DATA_TOTAL // NO_DATA
    NO_TRAIN_SAMPLE = cfg.exp.no_train_sample
    if NO_TRAIN_SAMPLE != "all":
        # Randomly select subsets of data to save time
        sample_idx = np.random.choice(sample_idx, size = NO_TRAIN_SAMPLE, replace = False)
    else:
        sample_idx = range(NO_DATA_TOTAL // NO_DATA)
        
    for idx, i in tqdm(enumerate(sample_idx)):
    # for idx, i in enumerate([14]):
        
        # Iterate over each sample
        print('\n\n======= Budget ratio: ', BUDGET_RATIO, ' and input budget ratio: ', BUDGET_RATIO_INPUT, ' and forecast budget ratio: ', BUDGET_RATIO_FORECAST, ' Sample idx: ', i, ' Train idx: ', idx, '======')
        
        # Use the NN extracted feature as the input of the ABF model
        feature = nn_extracted[i * NO_DATA:(i + 1) * NO_DATA]
        # True load and solar
        load = load_total[i * NO_DATA:(i + 1) * NO_DATA]
        solar = solar_total[i * NO_DATA:(i + 1) * NO_DATA]
        
        # Test the multi-uncertainty for ABF and OBF models
        
        print('==== Training the ABF model ====')
        # Train a full linear layer on the previous linear layer's output
        Wsolar_acc, bsolar_acc = train_abf_model(
                    feature, solar, SOLAR_MIN_CLIP * 1.001, SOLAR_MAX, 
                    reduced = False, verbose = VERBOSE)
        
        print('==== Training the obj model ====')
        start_time = time.time()
        Wsolar_obj, bsolar_obj, cost_obj_train = train_obj_kkt_model_reduced(
                        feature, load, solar, uc.prob_cvxpy, rd.prob_cvxpy,
                        uc.no_gen,
                        SOLAR_MIN_CLIP * 1.001, SOLAR_MAX,
                        Wsolar_acc = None, bsolar_acc = None,
                        verbose = VERBOSE, M=M_DP, alpha=ALPHA, 
                        # The dispatch problem becomes the kkt condition and linearized
                        reduced = False
                        )
        print('cost_obj_train: ', cost_obj_train)
        obj_time = time.time() - start_time
        
        # Function handles
        # No uncertainty
        evaluate_opt_func = partial(evaluate_opt, load = load, solar = solar, 
                                    uc_cvxpy = uc.prob_cvxpy, rd_cvxpy = rd.prob_cvxpy, 
                                    rd_class = rd)
        # Load/Opt Uncertainty
        worst_uncertainty_func = partial(worst_uncertainty, load = load, solar = solar, 
                                        uc_cvxpy = uc.prob_cvxpy, rd_cvxpy = rd.prob_cvxpy, 
                                        M_RD = M_RD, rd_class = rd, 
                                        budget_ratio = BUDGET_RATIO, verbose = VERBOSE)
        # Feature/Opt Uncertainty
        worst_uncertainty_func_multi = partial(worst_uncertainty_multi, 
                                        feature = feature, solar = solar, load = load, 
                                        uc_cvxpy = uc.prob_cvxpy, rd_cvxpy = rd.prob_cvxpy, 
                                        M_DP = M_DP, M_RD = M_RD, rd_class = rd, 
                                        budget_ratio = BUDGET_RATIO, 
                                        budget_ratio_input = BUDGET_RATIO_INPUT, 
                                        budget_ratio_forecast = BUDGET_RATIO_FORECAST,
                                        solar_min_clip = SOLAR_MIN_CLIP * 1.001, solar_max = SOLAR_MAX,
                                        verbose = VERBOSE,
                                        )
        
        print('==== ABF Performance ====')
        solar_pred_acc = evaluate_forecast(Wsolar_acc, bsolar_acc, feature, solar)
        cost_acc = evaluate_opt_func(solar_forecast=solar_pred_acc)['total_cost']
        worst_cost_acc = worst_uncertainty_func(solar_forecast=solar_pred_acc)
        worst_cost_acc_multi = worst_uncertainty_func_multi(Wsolar=Wsolar_acc, bsolar=bsolar_acc, 
                                                            input_uncertainty_only = False)
        worst_cost_acc_input = worst_uncertainty_func_multi(Wsolar=Wsolar_acc, bsolar=bsolar_acc, 
                                                            input_uncertainty_only = True)
        
        print('==== OBF Performance ====')
        solar_pred_obj = evaluate_forecast(Wsolar_obj, bsolar_obj, feature, solar)
        cost_obj = evaluate_opt_func(solar_forecast=solar_pred_obj)['total_cost']
        worst_cost_obj = worst_uncertainty_func(solar_forecast=solar_pred_obj)
        worst_cost_obj_multi = worst_uncertainty_func_multi(Wsolar=Wsolar_obj, bsolar=bsolar_obj,
                                                            input_uncertainty_only = False)
        worst_cost_obj_input = worst_uncertainty_func_multi(Wsolar=Wsolar_obj, bsolar=bsolar_obj,
                                                            input_uncertainty_only = True)
        
        def concat_or_init(new_data, dict_key):
            if idx == 0:
                return new_data
            else:
                if isinstance(performance_dict[dict_key], dict):
                    return {k: np.concatenate((performance_dict[dict_key][k], new_data[k]), axis=0) for k in performance_dict[dict_key].keys()}
                else:
                    return np.concatenate((performance_dict[dict_key], new_data), axis=0)
        
        # save the results
        performance_dict = {
            'solar': concat_or_init(solar, 'solar'),
            'load': concat_or_init(load, 'load'),
            'solar_pred_acc': concat_or_init(solar_pred_acc, 'solar_pred_acc'),
            'solar_pred_obj': concat_or_init(solar_pred_obj, 'solar_pred_obj'),
            'cost_acc': concat_or_init(cost_acc, 'cost_acc'),
            'cost_obj': concat_or_init(cost_obj, 'cost_obj'),
            'worst_cost_acc': concat_or_init(worst_cost_acc, 'worst_cost_acc'),
            'worst_cost_obj': concat_or_init(worst_cost_obj, 'worst_cost_obj'),
            'worst_cost_acc_multi': concat_or_init(worst_cost_acc_multi, 'worst_cost_acc_multi'),
            'worst_cost_acc_input': concat_or_init(worst_cost_acc_input, 'worst_cost_acc_input'),
            'worst_cost_obj_multi': concat_or_init(worst_cost_obj_multi, 'worst_cost_obj_multi'),
            'worst_cost_obj_input': concat_or_init(worst_cost_obj_input, 'worst_cost_obj_input'),
            'obj_time': concat_or_init([obj_time], 'obj_time'),
        }
        
    np.save(SAVING_DIR + f'{BUDGET_RATIO}_{BUDGET_RATIO_INPUT}_{BUDGET_RATIO_FORECAST}.npy', performance_dict, allow_pickle=True)
    
    print('======= Done for budget ratio: ', BUDGET_RATIO, ' and input budget ratio: ', BUDGET_RATIO_INPUT, '======')

if __name__ == '__main__':
    main()