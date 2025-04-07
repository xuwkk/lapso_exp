"""
A full implementation of the small-signal stability-constrained case study in the paper
"""

import sys
sys.path.append("./")
from lapso.neuralnet import form_milp
import torch
import hydra
from omegaconf import DictConfig
from pso.prepare import prepare_grid_from_pypower
from pso.operation import UC_DISCRETE
from pso.data import get_dataset_np
import numpy as np
import cvxpy as cp
from torch import nn
import os
from copy import deepcopy
from tqdm import tqdm
import time
from paper_exp.sco_func import (SmallSignalStability, 
                                train_logistic_assessor, 
                                return_nn, 
                                train_nn,
                                evaluate_uc)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    type = cfg.exp.type
    
    print(f"==========SCO experiment model (loss) type: {type}==========")

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Configuration
    xd = np.array(cfg.exp.xd)
    gscr_threshold = cfg.exp.gscr_threshold
    solar_min_clip = cfg.grid.renewable_min / cfg.grid.baseMVA   # minimum solar power to be considered
    over_value = cfg.exp.over_value           # margin for classifying stable as unstable
    solar_sample_no = cfg.exp.solar_sample_no # how many points when sampling each solar power
    solar_sample_ratio_list = cfg.exp.solar_sample_ratio_list
    train_config = cfg.exp.train_config       # training configuration
    # Threads = cfg.exp.Threads          # number of threads for gurobi
    # ConcurrentMIP = cfg.exp.ConcurrentMIP   # number of concurrent MIPs
    saving_dir = cfg.exp.saving_dir + f"{type}/"
    os.makedirs(saving_dir, exist_ok=True)
    T = cfg.operation.T
    renewable_rescale = cfg.exp.renewable_rescale
    start_date = cfg.exp.start_date
    end_date = cfg.exp.end_date
    assert cfg.operation.with_binary, "Consider the binary case for SCO study"
    
    # Grid
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    
    # Load data
    _, load_total, solar_total, _ = get_dataset_np(cfg.grid)
    solar_total = solar_total * renewable_rescale # Increase the solar power
    
    NO_DATA_TOTAL = load_total.shape[0]
    SOLAR_MAX = np.max(solar_total, axis=0)
    np.save(saving_dir + 'load_total.npy', load_total)
    np.save(saving_dir + 'solar_total.npy', solar_total)
    
    # Load optimization models
    uc = UC_DISCRETE(grid_xlsx, cfg.operation)
    uc.formulate()
    
    # Test if generation + renewable capacity is enough to cover the load
    print('ug_init: ', uc.ug_init)
    pg_max = (uc.pmax * uc.ug_init).sum()
    gen_diff = pg_max + np.sum(solar_total, axis = 1) - load_total.sum(axis = 1)
    print('gen_diff min: ', np.min(gen_diff), 'gen_diff max: ', np.max(gen_diff))
    assert np.all(gen_diff >= 0), "Generation + renewable capacity is not enough to cover the load"
    
    # Test solar penetration
    solar_per_day = np.sum(solar_total, axis = 1)
    load_per_day = np.sum(load_total, axis = 1)
    solar_per_day_ratio = solar_per_day / load_per_day
    print('solar_per_day_ratio min: ', 
          np.min(solar_per_day_ratio), 
          'solar_per_day_ratio max: ', 
          np.max(solar_per_day_ratio), 
          'solar_per_day_ratio mean: ', np.mean(solar_per_day_ratio))
    
    # Small signal stability analysis
    print('====== Small signal stability analysis ======')
    small_signal_stability = SmallSignalStability(uc, xd, gscr_threshold, solar_min_clip, SOLAR_MAX)
    
    # output_space: gSCR
    input_space, output_space = small_signal_stability.gen_dataset_small_signal(
                            solar_sample_no, solar_sample_ratio_list=solar_sample_ratio_list, verbose = True)
    
    print('input space: ', input_space.shape, 'output space: ', output_space.shape)
    print('Input space one example:', input_space[0])
    print('Output space one example:', output_space[0])
    
    print('===Train classifier===')
    label = np.zeros(output_space.shape[0])
    label[output_space <= gscr_threshold + over_value] = 1  # 1 for unstable
    print(f"Number of stable cases: {np.sum(label == 0)}, Number of unstable cases: {np.sum(label == 1)}")

    test_sample_idx = np.arange(0, NO_DATA_TOTAL, T)
    test_sample_idx = test_sample_idx[start_date:end_date]
    print(f"No of days to test: {test_sample_idx.shape[0]}")
    
    if 'linear' in type:
        # train the linear classifier
        W, b = train_logistic_assessor(input_space, label, type)
        W = torch.from_numpy(W[None,:]).float()
        b = torch.from_numpy(np.array([b])).float()
        print('Linear classifier trained.', 'W:', W.shape, 'b:', b.shape)
        # Convert to nn (not necessary)
        classifier = nn.Linear(input_space.shape[1], 1)
        classifier.weight.data = W
        classifier.bias.data = b
    else:
        # NN-based classifier
        classifier = return_nn(input_space.shape[1], type)
        
        model_path = os.path.join(saving_dir, f"{type}_classifier.pth")
        
        if os.path.exists(model_path):  
            # classifier = torch.load(model_path)
            classifier.load_state_dict(torch.load(model_path))
            print("Load the pre-trained classifier. If you want to train the classifier from scratch, please delete the file: ", 
                  model_path)
        else:
            classifier = train_nn(classifier, 
                                  torch.from_numpy(input_space).float(), 
                                  torch.from_numpy(label).float(), 
                                  **train_config)   
            torch.save(classifier.state_dict(), os.path.join(saving_dir, f"{type}_classifier.pth"))
    
    no_param = 0
    for param in classifier.parameters():
        if param.requires_grad:
            no_param += param.numel()
    print(f"Number of parameters of {type} classifier: {no_param}")
    
    print('===Test the classifier on the synthetic data===')
    test_input = torch.from_numpy(input_space).float()
    test_label = torch.from_numpy(label).float()
    test_output = classifier(test_input)
    test_predictions = (test_output.flatten() > 0).float()
    test_acc = (test_predictions == test_label).float().mean()
    print(f"Test accuracy: {test_acc.item()}")
    
    # Solve the original problem
    start_time = time.time()
    print('===Solve the original problem===')
    cost_ori, gscr_ori, gscr_cls_ori, ug_ori, solarc_ori, solar_ori, ls_ori = evaluate_uc(
        uc.prob_cvxpy, small_signal_stability, classifier, 
        test_sample_idx, load_total, solar_total, T, 
        threads = None, concurrent = None, seed = cfg.random_seed
        )
    time_ori = time.time() - start_time
    print(f"Time for solving the original problem: {time_ori / test_sample_idx.shape[0]}")
    
    no_binary_var_ori = 0
    for var in uc.prob_cvxpy.variables():
        if var.attributes['boolean']:
            no_binary_var_ori += np.prod(var.shape)
    print(f"Number of binary variables in the original problem: {no_binary_var_ori}")
    
    np.save(os.path.join(saving_dir, "ori_result.npy"), {
        'cost': cost_ori,
        'gscr': gscr_ori,
        'gscr_cls': gscr_cls_ori,
        'ug': ug_ori,
        'solarc': solarc_ori,
        'solar': solar_ori,
        'ls': ls_ori,
        'time': time_ori / test_sample_idx.shape[0],
        'no_binary_var': no_binary_var_ori
    }, allow_pickle=True)
    
    print('===Convert the original problem to the SCO problem===')
    
    # Extract relevant variable, parameter, and constraint
    original_prob = uc.prob_cvxpy
    ug = original_prob.var_dict['ug']       # (T,no_gen)
    sc = original_prob.var_dict['solarc']   # (T,no_solar)
    constraints = original_prob.constraints
    solar_as_parameter = original_prob.param_dict['solar']
    # Range of the variable: 
    # ug is bounded by 0 and 1, sc is bounded by 0 and solar_max - solar_min_clip
    # The bound is very loose
    lower_bound = np.zeros(uc.no_gen + uc.no_solar)
    upper_bound = np.concatenate([np.ones(uc.no_gen), SOLAR_MAX - solar_min_clip])
    initial_bound = (lower_bound[None,:], upper_bound[None,:])  
    
    # For each time step, add stability constraint
    for t in range(T):
        cls_constraint, (z,v) = form_milp(deepcopy(classifier), initial_bound, verbose = False)
        constraints.extend(cls_constraint)  # nn as MIL constraint
        constraints.extend([z[-1] <= -1e-3])    # small signal stability constraint
        constraints.extend(
            [z[0] == cp.hstack([ug[t], solar_as_parameter[t] - sc[t]])]
            # the constraint on sc has been included in the original problem
            ) # link the stability constraint to the original problem (NN input is related to decision variable)

    sco_prob = cp.Problem(original_prob.objective, constraints) # use the original objective
    
    print('===Solve the SCO problem===')
    start_time = time.time()
    # Solve the problem and evaluate the gscr and data gscr performance
    cost_sco, gscr_sco, gscr_cls_sco, ug_sco, solarc_sco, solar_sco, ls_sco = evaluate_uc(
        sco_prob, small_signal_stability, classifier, test_sample_idx, load_total, solar_total, T, 
        threads = None, concurrent = None, seed = cfg.random_seed)    
    time_sco = time.time() - start_time
    print(f"Time for solving the SCO problem: {time_sco / test_sample_idx.shape[0]}")
    
    no_binary_var_sco = 0
    for var in sco_prob.variables():
        if var.attributes['boolean']:
            no_binary_var_sco += np.prod(var.shape)
    print(f"Number of binary variables in the SCO problem: {no_binary_var_sco}")
    
    no_trainable_param = 0
    for param in classifier.parameters():
        if param.requires_grad:
            no_trainable_param += param.numel()
    print(f"Number of trainable parameters in the classifier: {no_trainable_param}")
    
    np.save(os.path.join(saving_dir, "sco_result.npy"), {
        'cost': cost_sco,
        'gscr': gscr_sco,
        'gscr_cls': gscr_cls_sco,
        'ug': ug_sco,
        'solarc': solarc_sco,
        'solar': solar_sco,
        'ls': ls_sco,
        'time': time_sco / test_sample_idx.shape[0],
        'no_binary_var': no_binary_var_sco,
        'no_trainable_param': no_trainable_param
    }, allow_pickle=True)
    
if __name__ == "__main__":
    main()