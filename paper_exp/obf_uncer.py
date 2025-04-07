"""
the full implemenration of the case study on objective-based forecast in the paper
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
                                ccg
                                )
from functools import partial
import torch
from paper_exp.train_abf_nn import MLP
from tqdm import tqdm
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
    BUDGET_RATIO = cfg.exp.budget_ratio
    SAVING_DIR = cfg.exp.saving_dir
    MAX_ITER = cfg.exp.max_iter
    TOL = cfg.exp.tol
    os.makedirs(SAVING_DIR, exist_ok=True)
    
    assert cfg.operation.with_binary == False, "Only consider continuous case for now"
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    
    # Load data
    feature_total, load_total, solar_total, _ = get_dataset_np(cfg.grid)
    NO_DATA_TOTAL = feature_total.shape[0]
    SOLAR_MAX = np.max(solar_total, axis = 0)
    
    # save the data
    # np.save(SAVING_DIR + 'feature_total.npy', feature_total)
    # np.save(SAVING_DIR + 'load_total.npy', load_total)
    # np.save(SAVING_DIR + 'solar_total.npy', solar_total)
    
    # Load NN
    model = MLP(70, 4)
    model.load_state_dict(torch.load(cfg.exp.train_config.nn_dir))
    model.to('cpu').eval()
    
    # Load optimization models
    uc = UC_CONTINUOUS(grid_xlsx, cfg.operation)
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
    nn_forecast, nn_extracted = model(torch.from_numpy(feature_total).float())
    nn_extracted = nn_extracted.detach().numpy()
    
    for i in tqdm(range(NO_DATA_TOTAL // NO_DATA)):
        
        # Use the NN extracted feature as the input of the ABF model
        feature = nn_extracted[i * NO_DATA:(i + 1) * NO_DATA]
        load = load_total[i * NO_DATA:(i + 1) * NO_DATA]
        solar = solar_total[i * NO_DATA:(i + 1) * NO_DATA]
        
        print('==== Training the ABF model ====')
        # Train a full linear layer on the previous linear layer
        Wsolar_acc, bsolar_acc = train_abf_model(
                    feature, solar, SOLAR_MIN_CLIP * 1.001, SOLAR_MAX, 
                    reduced = False, verbose = VERBOSE)
        
        print('==== Training the obj model ====')
        Wsolar_obj, bsolar_obj, cost_obj_train = train_obj_kkt_model_reduced(
                        feature, load, solar, uc.prob_cvxpy, rd.prob_cvxpy,
                        uc.no_gen,
                        SOLAR_MIN_CLIP * 1.001, SOLAR_MAX,
                        Wsolar_acc = None, bsolar_acc = None,
                        verbose = VERBOSE, M=M_DP, alpha=ALPHA, # The dispatch problem becomes the kkt condition and linearized
                        reduced = False
                        )
        
        print('==== Training the robust model ====')
        W_robust, b_robust, converged_obj = ccg(
            feature, load, solar,
            uc.prob_cvxpy, rd.prob_cvxpy, uc.no_gen,
            SOLAR_MIN_CLIP * 1.001, SOLAR_MAX,
            BUDGET_RATIO,
            reduced = False,
            Wsolar_acc = None, bsolar_acc = None,
            alpha = ALPHA, M_DP = M_DP, M_RD = M_RD, 
            max_iter = MAX_ITER, tol = TOL, verbose = VERBOSE)
        
        print('\n')    
        # Function handles
        evaluate_opt_func = partial(evaluate_opt, load = load, solar = solar, 
                                    uc_cvxpy = uc.prob_cvxpy, rd_cvxpy = rd.prob_cvxpy, 
                                    rd_class = rd)
        worst_uncertainty_func = partial(worst_uncertainty, load = load, solar = solar, 
                                        uc_cvxpy = uc.prob_cvxpy, rd_cvxpy = rd.prob_cvxpy, M_RD = M_RD, rd_class = rd, 
                                        budget_ratio = BUDGET_RATIO, verbose = VERBOSE)
        
        print('==== True Performance ====')
        cost_true = evaluate_opt_func(solar_forecast=solar)['total_cost']
        worst_cost_true = worst_uncertainty_func(solar_forecast=solar)
        
        print('==== ABF Performance ====')
        solar_pred_acc = evaluate_forecast(Wsolar_acc, bsolar_acc, feature, solar)
        cost_acc = evaluate_opt_func(solar_forecast=solar_pred_acc)['total_cost']
        worst_cost_acc = worst_uncertainty_func(solar_forecast=solar_pred_acc)
        
        print('==== OBF Performance ====')  
        solar_pred_obj = evaluate_forecast(Wsolar_obj, bsolar_obj, feature, solar)
        cost_obj = evaluate_opt_func(solar_forecast=solar_pred_obj)['total_cost']
        worst_cost_obj = worst_uncertainty_func(solar_forecast=solar_pred_obj)
        assert np.isclose(np.mean(cost_obj), cost_obj_train), "The cost of the objective-based forecast is not close to its training cost"
        
        print('==== CCG Performance ====')
        solar_pred_robust = evaluate_forecast(W_robust, b_robust, feature, solar)
        cost_robust = evaluate_opt_func(solar_forecast=solar_pred_robust)['total_cost']
        worst_cost_robust = worst_uncertainty_func(solar_forecast=solar_pred_robust)
        
        print('worst_cost_robust: ', np.mean(worst_cost_robust), 'converged_obj: ', converged_obj)
        assert np.isclose(np.mean(worst_cost_robust), converged_obj, atol=1e-3), "The cost of ccg does not match, may be caused by small M"
        
        # Helper function to concatenate data
        def concat_or_init(new_data, dict_key):
            if i == 0:
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
            'solar_pred_robust': concat_or_init(solar_pred_robust, 'solar_pred_robust'),
            'cost_true': concat_or_init(cost_true, 'cost_true'),
            'cost_acc': concat_or_init(cost_acc, 'cost_acc'),
            'cost_obj': concat_or_init(cost_obj, 'cost_obj'),
            'cost_robust': concat_or_init(cost_robust, 'cost_robust'),
            'worst_cost_true': concat_or_init(worst_cost_true, 'worst_cost_true'),
            'worst_cost_acc': concat_or_init(worst_cost_acc, 'worst_cost_acc'),
            'worst_cost_obj': concat_or_init(worst_cost_obj, 'worst_cost_obj'),
            'worst_cost_robust': concat_or_init(worst_cost_robust, 'worst_cost_robust'),
        }
        
    np.save(SAVING_DIR + f'{BUDGET_RATIO}.npy', performance_dict, allow_pickle=True)

    print('======= Done for budget ratio: ', BUDGET_RATIO, '======')
    
    
if __name__ == '__main__':
    main()
    
