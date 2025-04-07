"""
the implemenration of the case study on obf/basic in the paper
"""
import sys
sys.path.append('./')
import hydra
from omegaconf import DictConfig
from pso import prepare_grid_from_pypower, get_dataset_np, UC_CONTINUOUS, RD
import numpy as np
import os
from tqdm import tqdm
import torch
from paper_exp.train_abf_nn import MLP
from paper_exp.obf_func import (data_preprocess, 
                                train_abf_model, 
                                train_obj_kkt_model_reduced,
                                evaluate_forecast,
                                evaluate_opt
                                )

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    # to have LS work properly, the number of data should be large than the number of features
    NO_DATA = cfg.exp.train_config.no_data         # Time steps window
    M = cfg.exp.train_config.M                     # Big-M used for the KKT reformulation
    ALPHA = cfg.exp.train_config.alpha             # whether to include the penalization to the ABF-based model
    VERBOSE = cfg.exp.train_config.verbose
    SOLAR_MIN_CLIP = cfg.grid.renewable_min / cfg.grid.baseMVA
    print('SOLAR_MIN_CLIP: ', SOLAR_MIN_CLIP)
    SAVING_DIR = cfg.exp.saving_dir
    os.makedirs(SAVING_DIR, exist_ok=True)
    
    assert cfg.operation.with_binary == False, "Only consider continuous case for now"
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    
    # Load data
    feature_total, load_total, solar_total, _ = get_dataset_np(cfg.grid)
    NO_DATA_TOTAL = feature_total.shape[0]
    SOLAR_MAX = np.max(solar_total, axis = 0)  # Use the max solar power as the upper bound of the solar power forecast
    # save the data
    # np.save(SAVING_DIR + 'feature_total.npy', feature_total)
    # np.save(SAVING_DIR + 'load_total.npy', load_total)
    # np.save(SAVING_DIR + 'solar_total.npy', solar_total)
    
    print('feature shape: ', feature_total.shape, 'load shape: ', 
          load_total.shape, 'solar shape: ', solar_total.shape)
    
    # Load NN
    model = MLP(70, 4)
    model.load_state_dict(torch.load(cfg.exp.train_config.nn_dir))
    model.to('cpu').eval()
    
    # Load optimization models
    uc = UC_CONTINUOUS(grid_xlsx, cfg.operation)
    uc.formulate()
    rd = RD(grid_xlsx, cfg.operation)
    rd.formulate(discrete_uc=False)
    
    # test if generation + renewable capacity is enough to cover the load
    print('ug_init: ', uc.ug_init)
    pg_max = (uc.pmax * uc.ug_init).sum()
    gen_diff = pg_max + np.sum(solar_total, axis = 1) - load_total.sum(axis = 1)
    print('gen_diff min: ', np.min(gen_diff), 'gen_diff max: ', np.max(gen_diff))
    assert np.all(gen_diff >= 0), "Generation + renewable capacity is not enough to cover the load"
    
    # Preprocess the whole data
    feature_total, load_total, solar_total = data_preprocess(feature_total, load_total, solar_total)
    nn_forecast, nn_extracted = model(torch.from_numpy(feature_total).float())
    nn_extracted = nn_extracted.detach().numpy()
    
    for i in tqdm(range(NO_DATA_TOTAL // NO_DATA)):
        
        # Use the NN extracted feature as the input of the ABF model
        feature = nn_extracted[i * NO_DATA:(i + 1) * NO_DATA]
        load = load_total[i * NO_DATA:(i + 1) * NO_DATA]
        solar = solar_total[i * NO_DATA:(i + 1) * NO_DATA]
        
        Wsolar_acc, bsolar_acc = train_abf_model(
                    feature, solar, SOLAR_MIN_CLIP, SOLAR_MAX, 
                    reduced = False, verbose = VERBOSE)
        
        Wsolar_obj, bsolar_obj, cost_obj_train = train_obj_kkt_model_reduced(
                        feature, load, solar, uc.prob_cvxpy, rd.prob_cvxpy,
                        uc.no_gen,
                        SOLAR_MIN_CLIP, SOLAR_MAX,
                        Wsolar_acc, bsolar_acc,
                        verbose = VERBOSE, M=M, alpha=ALPHA,
                        reduced = False
                        )
        
        print('==== True Performance ====')
        cost_true = evaluate_opt(solar_forecast=solar, solar=solar, load=load, 
                                uc_cvxpy=uc.prob_cvxpy, rd_cvxpy=rd.prob_cvxpy, rd_class=rd)['total_cost']
        
        print('==== Acc Performance ====')
        solar_pred_acc = evaluate_forecast(Wsolar_acc, bsolar_acc, feature, solar)
        cost_acc = evaluate_opt(solar_pred_acc, solar, load, uc.prob_cvxpy, rd.prob_cvxpy, rd)['total_cost']
        
        print('==== Obj Performance ====')  
        solar_pred_obj = evaluate_forecast(Wsolar_obj, bsolar_obj, feature, solar)
        cost_obj = evaluate_opt(solar_pred_obj, solar, load, uc.prob_cvxpy, rd.prob_cvxpy, rd)['total_cost']
        
        # Verify kkt formulation is correct to the direct optimization
        assert np.isclose(np.mean(cost_obj), cost_obj_train), "The cost of the objective-based forecast is not close to its training cost"
        
        # save the results
        performance_dict = {
            'solar': solar if i == 0 else np.concatenate((performance_dict['solar'], solar), axis = 0),
            'load': load if i == 0 else np.concatenate((performance_dict['load'], load), axis = 0),
            'cost_true': cost_true if i == 0 else np.concatenate((performance_dict['cost_true'], cost_true), axis = 0),
            'cost_abf': cost_acc if i == 0 else np.concatenate((performance_dict['cost_abf'], cost_acc), axis = 0),
            'cost_obf': cost_obj if i == 0 else np.concatenate((performance_dict['cost_obf'], cost_obj), axis = 0),
            'solar_true': solar if i == 0 else np.concatenate((performance_dict['solar_true'], solar), axis = 0),
            'load_true': load if i == 0 else np.concatenate((performance_dict['load_true'], load), axis = 0),
            'solar_abf': solar_pred_acc if i == 0 else np.concatenate((performance_dict['solar_abf'], solar_pred_acc), axis = 0),
            'solar_obf': solar_pred_obj if i == 0 else np.concatenate((performance_dict['solar_obf'], solar_pred_obj), axis = 0),
        }
        
    np.save(SAVING_DIR + 'performance_dict.npy', performance_dict, allow_pickle=True)

if __name__ == '__main__':
    main()