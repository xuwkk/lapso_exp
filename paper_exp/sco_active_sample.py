"""
Run SCO experiments with active sampling strategy for dataset generation.
Actively sample data points around the gSCR boundary.
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
import os
from copy import deepcopy
import time
from tqdm import tqdm
from paper_exp.sco_func import (SmallSignalStability, 
                                return_nn, 
                                train_nn,
                                evaluate_uc,
                                evaluate_uc_basic)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Grid
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    grid_name = cfg.grid.data_dir.split('/')[-2]

    no_gen = len(grid_xlsx['gen'])
    no_renew = len(grid_xlsx['solar'])

    # Random generate the xd for all generators
    xd_min = cfg.exp.xd_min
    xd_max = cfg.exp.xd_max
    xd = xd_min + (xd_max - xd_min) * np.random.rand(no_gen)
  
    GSCR_THRESHOLD = cfg.exp.gscr_threshold
    SOLAR_MIN_CLIP= cfg.grid.renewable_min / cfg.grid.baseMVA   # minimum solar power to be considered
    OVER_VALUE = cfg.exp.over_value                              # margin for classifying stable as unstable
    TRAIN_CONFIG = cfg.exp.train_config                          # training configuration

    # experiment optimization setting
    SOLVER = cfg.exp.optimization.solver
    SOLVER_OPTION = cfg.exp.optimization.option
    REGRESSION = cfg.exp.regression
    TYPE = cfg.exp.type
    NEW_DATASET_FLAG = cfg.exp.generate_new_dataset     # whether to generate a new dataset
    NEW_NN_FLAG = cfg.exp.train_new_nn                  # whether to train a new NN
    NEW_AUG_FLAG = cfg.exp.generate_new_aug             # whether to do new data augmentation

    # Config for active sampling
    ACTIVE_SAMPLE = cfg.exp.active_sample
    ACTIVE_LR = cfg.exp.active_lr                        # learning rate for active sampling
    ACTIVE_MAX_ITER = cfg.exp.active_max_iter            # active sampling iterations
    

    print("\n")
    print("====================================================================")
    print(f"==========SCO on {grid_name} with {TYPE}, {SOLVER}, regression={REGRESSION}==========")
    print("====================================================================")


    saving_dir_dataset = cfg.exp.saving_dir + grid_name
    saving_dir_nn = cfg.exp.saving_dir + grid_name + f"/{TYPE}/"
    saving_dir = cfg.exp.saving_dir + grid_name + f"/{TYPE}/{SOLVER}/"
    os.makedirs(saving_dir, exist_ok=True)

    T = cfg.operation.T
    renewable_rescale = cfg.exp.renewable_rescale
    no_sample = cfg.exp.no_sample                       # larger set (days) for training
    no_sample_test = cfg.exp.no_sample_test             # smaller set (days) for test
    assert cfg.operation.with_binary, "Consider the binary case for SCO study"

    # Load data
    _, load_total, solar_total, _ = get_dataset_np(cfg.grid)

    # NOTE: data clip has been implemented in prepare.py
    solar_total = np.clip(solar_total * renewable_rescale, a_min =SOLAR_MIN_CLIP, a_max = None) # rescale the solar power 
    no_data_total = load_total.shape[0]
    solar_max = np.max(solar_total, axis=0)
    print(f"Solar Max: {solar_max}")

    print(no_data_total, " data points in total")

    # Load optimization models
    uc = UC_DISCRETE(grid_xlsx, cfg.operation)
    uc.formulate()

    # Test if generation + renewable capacity is enough to cover the load
    # using the initial status as example
    pg_max = (uc.pmax * uc.ug_init).sum()
    gen_diff = pg_max + np.sum(solar_total, axis = 1) - load_total.sum(axis = 1)
    print('gen_diff min per hour: ', np.min(gen_diff), 'gen_diff max per hour: ', np.max(gen_diff))
    assert np.all(gen_diff >= 0), "Generation + renewable capacity is not enough to cover the load"
    
    # Test solar penetration
    solar_per_hour = np.sum(solar_total, axis = 1) # aggregate solar power per hour
    load_per_hour = np.sum(load_total, axis = 1)
    per_hour_ratio = solar_per_hour / load_per_hour
    print(
        'solar_to_load_per_hour_ratio:',
        '--min: ', np.min(per_hour_ratio), 
        '--max: ', np.max(per_hour_ratio), 
        '--mean: ', np.mean(per_hour_ratio)
        )

    """Construct the training and test dataset"""

    # Training: random sample by day
    day_idx = np.random.choice(no_data_total // 24, no_sample, replace=False)
    # The index of the starting hour of a day
    sample_idx = [day_idx_ * 24 for day_idx_ in day_idx]
    # Random pick test samples from the training samples
    test_sample_idx = np.random.choice(sample_idx, no_sample_test, replace=False).tolist()

    small_signal_stability = SmallSignalStability(uc, xd, GSCR_THRESHOLD, SOLAR_MIN_CLIP, solar_max)

    # Performance on the basic UC problem P_basic
    if not os.path.exists(saving_dir_dataset + "/train_summary.npy") or NEW_DATASET_FLAG:
        print("========== Optimizing the P_basic results for getting the dataset ==========")
        cost_basic, gscr_basic, ug_basic, solarc_basic, solar_basic, ls_basic = evaluate_uc_basic(
                        uc.prob_cvxpy, small_signal_stability, 
                        sample_idx, load_total, solar_total, T,
                        solver = SOLVER,
                        opt_params=SOLVER_OPTION
                        ) 
        # save Pbasic results into a dictionary
        train_summary = {
            "cost_basic": cost_basic,
            "gscr_basic": gscr_basic,
            "ug_basic": ug_basic,
            "solarc_basic": solarc_basic,
            "solar_basic": solar_basic,
            "ls_basic": ls_basic,
            "xd": xd,
            "sample_idx": sample_idx,
            "sample_test_idx": test_sample_idx
        }
        np.save(saving_dir_dataset + "/train_summary.npy", train_summary, allow_pickle=True)

    else:
        print("========== Load existing P_basic training results ==========")
        basic_operation_summary = np.load(saving_dir_dataset + "/train_summary.npy", allow_pickle=True).item()
        cost_basic = basic_operation_summary["cost_basic"]
        gscr_basic = basic_operation_summary["gscr_basic"]
        ug_basic = basic_operation_summary["ug_basic"]
        solarc_basic = basic_operation_summary["solarc_basic"]
        solar_basic = basic_operation_summary["solar_basic"]
        ls_basic = basic_operation_summary["ls_basic"]
        xd_prime = basic_operation_summary["xd"]
        assert np.allclose(xd, xd_prime), "xd not consistent!"
    
    print("Performance on the original P_basic problem")

    # Histogram
    counts, bin_edges = np.histogram(gscr_basic.flatten(), bins=10)
    for count, edge_start, edge_end in zip(counts, bin_edges[:-1], bin_edges[1:]):
        print(f"gSCR range [{edge_start:.2f}, {edge_end:.2f}): {count} samples")
    
    no_unstable = np.sum(gscr_basic < GSCR_THRESHOLD)
    no_stable_small = np.sum((gscr_basic >= GSCR_THRESHOLD) & (gscr_basic < GSCR_THRESHOLD + OVER_VALUE))
    print(f"No of unstable samples: {no_unstable}, no of stable but small margin samples: {no_stable_small}")

    ave_cost_basic = np.mean(cost_basic)
    gscr_violation_ratio_basic = np.sum(gscr_basic.reshape(-1) < GSCR_THRESHOLD) / len(gscr_basic.reshape(-1))
    print(f"Ave. cost (day basis) {ave_cost_basic:.2f}", 
            f"gSCR violation ratio (hour basis) {gscr_violation_ratio_basic:.4f}")
    ori_gen_on_sum = np.sum(ug_basic, axis = (0,1))
    print(f"No of on generators over the training samples: {ori_gen_on_sum}") # (no_gen,)

    ug_basic = ug_basic.reshape(-1, no_gen) # (no_sample * T, no_gen)
    solar_basic = solar_basic.reshape(-1, no_renew)
    solarc_basic = solarc_basic.reshape(-1, no_renew)
    
    """ Data augmentation by active sampling around the gSCR boundary """
    if ACTIVE_SAMPLE:
        if NEW_AUG_FLAG:
            print("========== Active sampling data augmentation ==========")
            
            ug_new, psolar_new = [], []
            for i in tqdm(range(len(ug_basic)), miniters=20):
                ug_i = ug_basic[i]
                psolar_i = solar_basic[i] - solarc_basic[i]  # use the net solar power

                # Perturb psolar
                ug_i_new, psolar_i_new = small_signal_stability.perturb_data(
                                                ug_i, psolar_i, 
                                                on_cost=uc.zero, 
                                                lr = ACTIVE_LR, max_iter = ACTIVE_MAX_ITER,
                                                verbose = False)
                ug_new.append(ug_i_new)
                psolar_new.append(psolar_i_new)

                # if i >= 20:
                #     break
            
            ug_new = np.vstack(ug_new)
            psolar_new = np.vstack(psolar_new)
            print("Shape of augmented data..")
            print(ug_new.shape, psolar_new.shape)

            ug_basic = np.vstack((ug_basic, ug_new))
            solar_basic = np.vstack((solar_basic, psolar_new))
            solarc_basic = np.vstack((solarc_basic, np.zeros_like(psolar_new))) # dummy solarc

            # Save the augmented dataset
            aug_train_dataset = {
                "ug_basic": ug_basic,
                "solarc_basic": solarc_basic,
                "solar_basic": solar_basic
            }

            np.save(saving_dir_dataset + "/aug_train_dataset.npy", aug_train_dataset, allow_pickle=True)

        else:
            print("========== Load existing augmented dataset ==========")
            aug_train_dataset = np.load(saving_dir_dataset + "/aug_train_dataset.npy", allow_pickle=True).item()
            ug_basic = aug_train_dataset["ug_basic"]
            solarc_basic = aug_train_dataset["solarc_basic"]
            solar_basic = aug_train_dataset["solar_basic"]

        print("Shape of augmented data..")
        print(ug_basic.shape, solar_basic.shape, solarc_basic.shape)

        # Evaluate the gSCR for the new and old data points
        gscr_basic_ori = deepcopy(gscr_basic).flatten()
        gscr_basic = []
        for i in range(len(ug_basic)):
            gscr_i = small_signal_stability.compute_gSCR(ug_basic[i], solar_basic[i] - solarc_basic[i])
            gscr_basic.append(gscr_i)
        
        gscr_basic = np.array(gscr_basic)
        print(gscr_basic.shape, gscr_basic_ori.shape)
        assert np.all(gscr_basic_ori == gscr_basic[:len(gscr_basic_ori)]), "gscr not consistent after augmentation"

        # Histogram of the gscr values
        counts, bin_edges = np.histogram(gscr_basic.flatten(), bins=10)
        print("Histogram of gSCR values in the training dataset:")
        for count, edge_start, edge_end in zip(counts, bin_edges[:-1], bin_edges[1:]):
            print(f"gSCR range [{edge_start:.2f}, {edge_end:.2f}): {count} samples")
        no_unstable = np.sum(gscr_basic < GSCR_THRESHOLD)
        no_stable_small = np.sum((gscr_basic >= GSCR_THRESHOLD) & (gscr_basic < GSCR_THRESHOLD + OVER_VALUE))
        print(f"No of unstable samples: {no_unstable}, no of stable but small margin samples: {no_stable_small}")

    """
    Build NN classifier: can be classification or regression
    """
    # Train NN as the stability assessor

    if not os.path.exists(saving_dir_nn + "/nn_model.pth") or NEW_NN_FLAG:
        print("Train new NN model")
        # Dataset ug: (no_day, T, no_gen) / solarc: (no_day, T, no_renew)
        input_feature = np.hstack((ug_basic, solar_basic - solarc_basic))

        print("input feature shape:", input_feature.shape)
        if REGRESSION: 
            # Directly forecast the gscr value
            label = gscr_basic.reshape(-1) 
        else:
            # Default
            # Classify stable and unstable cases: stable 0; unstable 1
            label = (gscr_basic.reshape(-1) < GSCR_THRESHOLD + OVER_VALUE).astype(int) # convervatibe labeling
            print(f"No of unstable samples: {np.sum(label)}, no of stable samples: {len(label) - np.sum(label)}")

        # Train NN
        classifier = return_nn(input_feature.shape[1], TYPE)
        classifier = train_nn(
                model = classifier, data = torch.from_numpy(input_feature).float(), 
                label = torch.from_numpy(label).float(),regression = REGRESSION,  **TRAIN_CONFIG
            )
        # save NN
        torch.save(classifier.state_dict(), saving_dir_nn + "/nn_model.pth")

    else:
        print("Load existing NN model")
        classifier = return_nn(no_gen + no_renew, TYPE)
        classifier.load_state_dict(torch.load(saving_dir_nn + "/nn_model.pth", weights_only=True))
    
    """Evaluate P_basic"""
    print("========== Evaluate on P_basic on the test days ==========")
    # These samples have been solved in training but we re-evaluate them here
    start_time = time.time()
    cost_ori, gscr_ori, gscr_cls_ori, ug_ori, solarc_ori, solar_ori, ls_ori = evaluate_uc(
        uc.prob_cvxpy, small_signal_stability, classifier, 
        test_sample_idx, load_total, solar_total, T, regression = REGRESSION,
        threads = None, concurrent = None, seed = cfg.random_seed,
        solver = SOLVER, opt_params = SOLVER_OPTION
        )
    time_ori = time.time() - start_time
    print(f"Time for solving the Pbasic on test dataset: {time_ori / len(test_sample_idx)}")
    
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
        'time': time_ori / len(test_sample_idx),
        'no_binary_var': no_binary_var_ori,
        "ori_gen_on_sum": np.sum(ug_ori, axis = (0,1))
    }, allow_pickle=True)

    """Performance of Pinf^sco"""
    
    # Encode into the UC problem as P_inf^sco
    print("========== Evaluate on P_sco on the test days ==========")
    lower_bound = np.zeros(uc.no_gen + uc.no_solar)

    original_prob = uc.prob_cvxpy
    ug = original_prob.var_dict['ug']       # (T,no_gen)
    sc = original_prob.var_dict['solarc']   # (T,no_solar)
    solar_as_parameter = original_prob.param_dict['solar']
    constraints = original_prob.constraints
    upper_bound = np.concatenate([np.ones(uc.no_gen), solar_max])  # ug \in {0,1}, sc \in [0, solar_max]
    initial_bound = (lower_bound[None,:], upper_bound[None,:])

    # For each time step, add stability constraint
    for t in range(T):
        cls_constraint, (z,v) = form_milp(deepcopy(classifier), initial_bound, verbose = False)

        # NN as MIL constraint
        constraints.extend(cls_constraint)  

        # Small signal stability constraint
        if REGRESSION:
            # directly satisfy gscr constraint (sligtly larger)
            constraints.extend([z[-1] >= GSCR_THRESHOLD + OVER_VALUE])  # TODO: make this larger
        else:
            constraints.extend([z[-1] <= -1])  # TODO: make this more negative 

        # Link the decision variable/parameter to the input of NN   
        constraints.extend(
            [z[0] == cp.hstack([ug[t], solar_as_parameter[t] - sc[t]])]
            # the constraint on sc has been included in the original problem
            ) # link the stability constraint to the original problem (NN input is related to decision variable)

    # Formulate P_inf^sco
    sco_prob = cp.Problem(original_prob.objective, constraints) # use the original objective
    
    # Solve the problem and evaluate the gscr and data gscr performance
    cost_sco, gscr_sco, gscr_cls_sco, ug_sco, solarc_sco, solar_sco, ls_sco = evaluate_uc(
        sco_prob, small_signal_stability, classifier, 
        test_sample_idx, load_total, solar_total, T, 
        regression = REGRESSION,
        threads = None, concurrent = None, seed = cfg.random_seed,
        solver = SOLVER, opt_params = SOLVER_OPTION,
        )    
    time_sco = time.time() - start_time
    print(f"Time for solving the SCO problem: {time_sco / len(test_sample_idx)}")
    
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

    sco_gen_on_sum = np.sum(ug_sco, axis = (0,1))
    print(f"No of on generators over the SCO samples: {sco_gen_on_sum}")

    # save the results
    np.save(os.path.join(saving_dir, "sco_result.npy"), {
        'cost': cost_sco,
        'gscr': gscr_sco,
        'gscr_cls': gscr_cls_sco,
        'ug': ug_sco,
        'solarc': solarc_sco,
        'solar': solar_sco,
        'ls': ls_sco,
        'time': time_sco / len(test_sample_idx),
        'no_binary_var': no_binary_var_sco,
        'no_trainable_param': no_trainable_param,
        "sco_gen_on_sum": sco_gen_on_sum

    }, allow_pickle=True)

if __name__ == "__main__":
    main()


