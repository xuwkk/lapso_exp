"""
Sensitivity analysis of ABF, OBF/Basic, and SCO
"""

"""
Objective-based forecast based on the SCO model
"""
import sys
sys.path.append('./')
import hydra
from omegaconf import DictConfig
from pso.data import get_dataset_np
from pso.operation import UC_CONTINUOUS, RD
import numpy as np
import os
import torch
from tqdm import tqdm
from paper_exp.train_abf_nn import MLP
from pso.prepare import prepare_grid_from_pypower
from paper_exp.sco_func import SmallSignalStability, train_logistic_assessor
from paper_exp.obf_func import (add_stability_constraint, train_obj_kkt_model_reduced,
                                evaluate_forecast, evaluate_opt,
                                return_grad,
                                train_abf_model,
                                data_preprocess)
from functools import partial
        
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    # to have LS work properly, the number of data should be large than the number of features
    NO_DATA = cfg.exp.train_config.no_data   # training time window
    M = cfg.exp.train_config.M
    ALPHA = cfg.exp.train_config.alpha
    VERBOSE = cfg.exp.train_config.verbose
    SOLAR_MIN_CLIP = cfg.grid.renewable_min / cfg.grid.baseMVA
    SAVING_DIR = cfg.exp.saving_dir + '/obf_sco_grad/'
    os.makedirs(SAVING_DIR, exist_ok=True)
    TYPE = cfg.exp.type
    XD = np.array(cfg.exp.xd)
    GSCR_THRESHOLD = cfg.exp.gscr_threshold
    OVER_VALUE = cfg.exp.over_value    # <= gscr+over_value is the threshold for unstable samples (more samples are assigned as unstable)
    SOLAR_SAMPLE_NO = cfg.exp.solar_sample_no  # number of evenly sampled solar power between min and max for training stability assessor
    assert cfg.operation.with_binary == False, "Only consider continuous case for now"
    
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    
    # Load data
    feature_total, load_total, solar_total, _ = get_dataset_np(cfg.grid)
    NO_DATA_TOTAL = feature_total.shape[0]
    SOLAR_MAX = np.max(solar_total, axis = 0)
    # # Save the data
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
    
    # Small signal stability analysis
    print('====== Small signal stability analysis ======')
    ug = uc.ug_init
    print('Fixed ug: ', ug)
    small_signal_stability = SmallSignalStability(uc, XD, GSCR_THRESHOLD, 
                                                SOLAR_MIN_CLIP, SOLAR_MAX)

    # Generate the input and output space for the stability assessment
    input_space, output_space = small_signal_stability.gen_dataset_small_signal(
                                    solar_sample_no = SOLAR_SAMPLE_NO, 
                                    fixed_ug = ug)
    
    print('input space: ', input_space.shape, 'output space: ', output_space.shape)
    print('input space one example:', input_space[0])
    print('output space one example:', output_space[0])
    
    print('====== Train a logistic regression model ======')
    # Train a logistic regression model to predict the stability of the system
    label = np.zeros(output_space.shape[0])
    label[output_space <= GSCR_THRESHOLD + OVER_VALUE] = 1  # 1 for unstable set more to be unstable
    print(f"Number of stable cases: {np.sum(label == 0)}, Number of unstable cases: {np.sum(label == 1)}")
    
    W_assessor, b_assessor = train_logistic_assessor(input_space, label, type = TYPE)
    
    print('W_assessor: ', W_assessor.shape, 'b_assessor: ', b_assessor.shape)
    
    # Add stability constraint to the UC and RD
    uc_ori = uc.prob_cvxpy
    rd_ori = rd.prob_cvxpy
    uc_sco = add_stability_constraint(uc_ori, W_assessor, b_assessor)
    rd_sco = add_stability_constraint(rd_ori, W_assessor, b_assessor)
    
    # Preprocess the whole dataset: mainly normalize the feature
    feature_total, load_total, solar_total = data_preprocess(feature_total, load_total, solar_total)
    nn_forecast, nn_extracted = model(torch.from_numpy(feature_total).float())
    nn_extracted = nn_extracted.detach().numpy()
    
    # # Lets test the small signal stability of the true solar data and the ug_init plan
    # print('====== Test the small signal stability of the true solar data and the ug_init plan ======')
    # print('ug_init: ', ug)
    # print('Stability performance of the True solar data: ')
    # gscr_true_list = []
    # for i in range(len(solar_total)):
    #     gscr = small_signal_stability.compute_gSCR(ug, solar_total[i])
    #     gscr_true_list.append(gscr)
    # unstable_idx = np.where(np.array(gscr_true_list) <= GSCR_THRESHOLD)[0]
    # print(f"Unstable ratio hourly: {len(unstable_idx)/len(gscr_true_list)}")
    # gscr_true_reshape = np.reshape(gscr_true_list, (len(solar_total) // 24, 24))
    # unstable_idx_daily = np.where(np.sum(gscr_true_reshape <= GSCR_THRESHOLD, axis=1) > 0)[0]
    # print(f"Unstable ratio daily: {len(unstable_idx_daily)/len(gscr_true_reshape)}")
    
    performance = {
        "cost_acc": [],
        "cost_obj": [],
        "cost_obj_sco": [],
        "grad_W_acc": [],
        "grad_b_acc": [],
        "grad_W_obj": [],
        "grad_b_obj": [],
        "grad_W_obj_sco": [],
        "grad_b_obj_sco": []
    }
    
    for i in tqdm(range(NO_DATA_TOTAL // NO_DATA)):
        
        # Use the NN extracted feature as the input of the ABF model
        feature = nn_extracted[i * NO_DATA:(i + 1) * NO_DATA]
        load = load_total[i * NO_DATA:(i + 1) * NO_DATA]
        solar = solar_total[i * NO_DATA:(i + 1) * NO_DATA]
        
        # print('==== Training the ABF model ====')
        # Train a full linear layer on the previous linear layer
        Wsolar_acc, bsolar_acc = train_abf_model(
                    feature, solar, SOLAR_MIN_CLIP * 1.001, SOLAR_MAX, 
                    reduced = False, verbose = VERBOSE)
        
        # print('==== Training the OBF/Basic model ====')
        Wsolar_obj, bsolar_obj, cost_obj_train = train_obj_kkt_model_reduced(
                        feature, load, solar, uc_ori, rd_ori,
                        uc.no_gen,
                        SOLAR_MIN_CLIP * 1.001, SOLAR_MAX,
                        Wsolar_acc = None, bsolar_acc = None,  # Normally we set the regularization alpha to be 0
                        verbose = VERBOSE, M=M, alpha=ALPHA,
                        reduced = False   
                        )
        
        # print('==== Training the OBF/SCO model ====')
        Wsolar_obj_sco, bsolar_obj_sco, cost_obj_sco_train = train_obj_kkt_model_reduced(
                        feature, load, solar, uc_sco, rd_sco, # pass the SCO model
                        uc.no_gen,
                        SOLAR_MIN_CLIP * 1.001, SOLAR_MAX,
                        Wsolar_acc = None, bsolar_acc = None,
                        verbose = VERBOSE, M=M, alpha=ALPHA,
                        reduced = False
                        )
        
        # Compute the gradient of the total cost with respect to the forecast weight (and bias)
        cost_acc, grad_W_acc, grad_b_acc, success_idx_acc = return_grad(Wsolar_acc, bsolar_acc, feature, load, solar, uc_ori, rd_ori, rd_class=rd)
        cost_obj, grad_W_obj, grad_b_obj, success_idx_obj = return_grad(Wsolar_obj, bsolar_obj, feature, load, solar, uc_ori, rd_ori, rd_class=rd)
        cost_obj_sco, grad_W_obj_sco, grad_b_obj_sco, success_idx_obj_sco = return_grad(Wsolar_obj_sco, bsolar_obj_sco, feature, load, solar, uc_ori, rd_ori, rd_class=rd)
        
        # Intersect the success index
        success_idx = np.intersect1d(success_idx_acc, success_idx_obj, success_idx_obj_sco)
        
        performance["cost_acc"].append(cost_acc[success_idx])
        performance["cost_obj"].append(cost_obj[success_idx])
        performance["cost_obj_sco"].append(cost_obj_sco[success_idx])
        
        performance["grad_W_acc"].append(grad_W_acc[success_idx])
        performance["grad_b_acc"].append(grad_b_acc[success_idx])
        performance["grad_W_obj"].append(grad_W_obj[success_idx])
        performance["grad_b_obj"].append(grad_b_obj[success_idx])
        performance["grad_W_obj_sco"].append(grad_W_obj_sco[success_idx])
        performance["grad_b_obj_sco"].append(grad_b_obj_sco[success_idx])

    # for key, value in performance.items():
    #     print(key, np.mean(value))
    
    # concatenate the performance
    for key, value in performance.items():
        performance[key] = np.concatenate(value, axis=0)
    
    grad_W_acc = performance["grad_W_acc"]
    grad_W_obj = performance["grad_W_obj"]
    grad_W_obj_sco = performance["grad_W_obj_sco"]
    grad_b_acc = performance["grad_b_acc"]
    grad_b_obj = performance["grad_b_obj"]
    grad_b_obj_sco = performance["grad_b_obj_sco"]
    
    grad_W_acc_norm = np.linalg.norm(grad_W_acc, axis=-1)
    grad_W_obj_norm = np.linalg.norm(grad_W_obj, axis=-1)
    grad_W_obj_sco_norm = np.linalg.norm(grad_W_obj_sco, axis=-1)
    grad_b_acc_norm = np.linalg.norm(grad_b_acc, axis=-1)
    grad_b_obj_norm = np.linalg.norm(grad_b_obj, axis=-1)
    grad_b_obj_sco_norm = np.linalg.norm(grad_b_obj_sco, axis=-1)
    
    print(np.mean(grad_W_acc_norm), np.mean(grad_W_obj_norm), np.mean(grad_W_obj_sco_norm))
    print(np.mean(grad_b_acc_norm), np.mean(grad_b_obj_norm), np.mean(grad_b_obj_sco_norm))
    
    # Compute the cosine similarity
    cos_sim_acc_obj = np.sum(grad_W_acc * grad_W_obj, axis=-1) / (grad_W_acc_norm * grad_W_obj_norm)                                                        
    cos_sim_obj_obj_sco = np.sum(grad_W_obj * grad_W_obj_sco, axis=-1) / (grad_W_obj_norm * grad_W_obj_sco_norm)
    
    print('cos_sim_acc_obj: ', len(cos_sim_acc_obj), np.mean(cos_sim_acc_obj))
    print('cos_sim_obj_obj_sco: ', len(cos_sim_obj_obj_sco), np.mean(cos_sim_obj_obj_sco))
    
    performance["cos_sim_acc_obj"] = cos_sim_acc_obj
    performance["cos_sim_obj_obj_sco"] = cos_sim_obj_obj_sco
    
    # Save the performance
    np.save(SAVING_DIR + 'performance.npy', performance, allow_pickle=True)
    
if __name__ == '__main__':
    main()