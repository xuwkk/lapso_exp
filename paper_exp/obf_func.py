"""
utility functions for the objective-based forecast case study in the paper
- abf
- obf-basic
- obf-sco
- obf-uncertain
"""

import numpy as np
import cvxpy as cp
from lapso.optimization import form_kkt, return_standard_form
from cvxpylayers.torch import CvxpyLayer
import torch


def data_preprocess(feature, load, solar):
    
    NO_DATA = feature.shape[0]
    
    # Normalize the feature data
    feature_mean = np.mean(feature, axis=(0,1))
    feature_std = np.std(feature, axis=(0,1))
    feature_mean[:4] = 0
    feature_std[:4] = 1
    
    # Apply normalization
    feature = (feature - feature_mean[None,None,:]) / feature_std[None,None,:]
    
    # Reshape and concatenate features
    feature_other = feature[:,:,4:].reshape(NO_DATA, -1)
    feature = np.concatenate([feature[:, 0, :4], feature_other], axis=1)
    
    return feature, load, solar
    
# def data_preprocess(feature, load, solar, NO_DATA, start_idx):

#     """
#     return data (subset)
#     normalized the feature data"""
    
#     sample_idx = np.arange(start_idx, start_idx + NO_DATA)
    
#     feature = feature[sample_idx]
#     load = load[sample_idx] 
#     solar = solar[sample_idx]
    
#     # Normalize features except calendar features (first 4)
#     feature_mean = np.mean(feature, axis=(0,1))
#     feature_std = np.std(feature, axis=(0,1))
#     feature_mean[:4] = 0
#     feature_std[:4] = 1
    
#     # Apply normalization
#     feature = (feature - feature_mean[None,None,:]) / feature_std[None,None,:]
    
#     # Reshape and concatenate features
#     feature_other = feature[:,:,4:].reshape(NO_DATA, -1)
#     feature = np.concatenate([feature[:, 0, :4], feature_other], axis=1)

#     return feature, load, solar

def train_abf_model(feature, solar, solar_min_clip, solar_max, reduced = False, verbose = False):
    """
    train the accuracy-based forecast model
    only forecast the solar"""
    
    # feature: batch_size x no_feature
    if reduced:
        # train a diagonal 
        assert feature.shape[1] == solar.shape[1]
        Wsolar_ = cp.Variable(shape = (feature.shape[1],))
        Wsolar = cp.diag(Wsolar_)
    else:
        # train a full linear regression model
        Wsolar = cp.Variable(shape = (feature.shape[1], solar.shape[1]))
    
    bsolar = cp.Variable(shape = (1, solar.shape[1])) # allow batch propagation
    solar_forecast = cp.matmul(feature, Wsolar) + bsolar
    solar_loss = cp.sum(cp.abs(solar_forecast - solar)) / feature.shape[0]
    constraints = [solar_forecast >= solar_min_clip, solar_forecast <= solar_max[None]]
    prob_solar = cp.Problem(cp.Minimize(solar_loss), constraints)
    prob_solar.solve(solver = cp.GUROBI, verbose = verbose)
    
    return Wsolar.value, bsolar.value

def train_obj_kkt_model_reduced(feature, load, solar, uc_cvxpy, rd_cvxpy, 
                                no_gen,
                                solar_min_clip,
                                solar_max,
                                Wsolar_acc, bsolar_acc,
                                alpha = 0.0,
                                M = 1e4,
                                reduced = False,
                                verbose = False):
    """
    Use the reduced form by moving the lower level RD as upper level constraints
    only forecast the solar
    """
    # feature: N x no_feature
    # Variables: feature @ Wsolar + bsolar
    if reduced:
        # train a diagonal matrix
        Wsolar_ = cp.Variable(shape = (feature.shape[1],))
        Wsolar = cp.diag(Wsolar_)
    else:
        Wsolar = cp.Variable(shape = (feature.shape[1], solar.shape[1])) # XW + b
    bsolar = cp.Variable(shape = (1, solar.shape[1]))
    
    # KKT of UC
    kkt_uc_constraints, kkt_uc_variable, uc_param,_ = form_kkt(uc_cvxpy, M)
    # UC objective that contributes to the UL objective
    P_uc = uc_param['P'][:no_gen, :no_gen]
    q_uc = uc_param['q'][:no_gen]
    
    # Standard form of RD (becomes the UL constraints)
    P_rd, q_rd, A_rd, G_rd, b_rd, h_rd, B_rd, H_rd = return_standard_form(rd_cvxpy)
    
    # Define variables for RD decision variables
    no_var_rd = P_rd.shape[0]
    z_rd_list = [cp.Variable(shape = (no_var_rd,)) for _ in range(feature.shape[0])]
    
    # Constraints
    constraints = []
    obj = 0
    
    # Forecast constraints in UL
    solar_forecast_all = feature @ Wsolar + bsolar  # XW + b [N, no_solar]
    constraints += [solar_forecast_all >= solar_min_clip, solar_forecast_all <= solar_max[None]]
        
    # # Dummy constraints to make the model feasible
    # constraints += [Wsolar == np.eye(solar.shape[1])]
    # constraints += [bsolar == np.zeros((1, solar.shape[1]))]
    
    for i in range(feature.shape[0]):
        # Foreacst horizon
        # Forecaster
        solar_forecast = solar_forecast_all[i] # [no_solar]
        load_true = load[i]
        solar_true = solar[i]
        
        # LL UC constraints
        # regenerate the UC kkt constraints again
        kkt_uc_constraints, kkt_uc_variable, uc_param,_ = form_kkt(uc_cvxpy, M)
        param_uc_var = kkt_uc_variable['param_dict_as_var']
        z_uc = kkt_uc_variable['x_dict']
        
        constraints.extend(kkt_uc_constraints)   # UC KKT constraints
        constraints+= [ # assign the forecast to the parameters (actually another variable)
            param_uc_var['load'] == load_true,
            param_uc_var['solar'] == solar_forecast,
            ]
        
        # UL RD constraints
        constraints+= [
            # rd constraints becomes the UL constraints
            A_rd @ z_rd_list[i] == b_rd + B_rd['load'] @ load_true + B_rd['solar'] @ solar_true + B_rd['pg_parameter'] @ z_uc['pg'],
            G_rd @ z_rd_list[i] <= h_rd + H_rd['load'] @ load_true + H_rd['solar'] @ solar_true + H_rd['pg_parameter'] @ z_uc['pg']
        ]
        
        # Objective
        uc_obj = 0.5 * cp.quad_form(z_uc['pg'], P_uc) + q_uc @ z_uc['pg']
        rd_obj = 0.5 * cp.quad_form(z_rd_list[i], P_rd) + q_rd @ z_rd_list[i]
    
        obj += uc_obj + rd_obj  
    
    obj_all = obj / feature.shape[0]
    
    # add the regularization term
    if alpha > 0:
        obj_all += alpha * (
            cp.mean(cp.abs(Wsolar - Wsolar_acc)) + cp.mean(cp.abs(bsolar - bsolar_acc)))
        # obj_all += alpha * (cp.sum(cp.abs(Wsolar)) + cp.sum(cp.abs(bsolar)))
        
    prob = cp.Problem(cp.Minimize(obj_all), constraints)
    prob.solve(solver = cp.GUROBI, verbose = verbose, 
            #    Threads = 100, Presolve = 2, 
            #    MIPFocus = 2, Cuts = 2, VarBranch = 2
               )
    
    # print(f'obj: {obj.value/feature.shape[0]}')
    # print(f'obj_all: {obj_all.value}')
    
    return Wsolar.value, bsolar.value, obj.value/feature.shape[0]

# Solve optimization problem
def assign_parameter(original_problem, original_input):
    for param in original_problem.parameters():
        try:
            param.value = original_input[param.name()]
        except:
            print(f"Parameter {param.name()} not found in original input or the shape mismatches")


def solve_problem(problem, input: dict, verbose = False, threads = 1, concurrent = 1, seed = 0):

    assign_parameter(problem, input)
    problem.solve(solver = cp.GUROBI, verbose = verbose, Threads = threads, ConcurrentMIP = concurrent, Seed = seed)
    solution = {var.name(): var.value for var in problem.variables()}
    optimal_value = problem.value
    
    return solution, optimal_value

def evaluate_forecast(W, b, feature, solar):
    
    pred = feature @ W + b
    mape = np.mean(np.abs(pred - solar) / solar)
    rmse = np.sqrt(np.mean(np.square(pred - solar)))
    print(f'Solar MAPE: {mape}')
    print(f'Solar RMSE: {rmse}')
    
    diff = pred.sum(axis=1) - solar.sum(axis=1)
    print('diff (positive means over-forecast): ', np.mean(diff), np.std(diff))
    
    return pred

def evaluate_opt(solar_forecast, solar, load, uc_cvxpy, rd_cvxpy, 
                 rd_class, 
                 # for stability assessment only
                 W_assessor = None, b_assessor = None, 
                 small_signal_stability = None,
                 # for uncertainty analysis only
                 load_forecast = None):
    """
    Evaluate the performance of the OBF model
    W_assessor, b_assessor, small_signal_stability: for stability assessment only
    load_forecast: for uncertainty analysis only
    """
    
    total_cost = []  
    if W_assessor is not None:
        uc_cls_result = [] # stability assessment result of UC based on the classifier
        rd_cls_result = []
        uc_gscr_result = [] # gSCR of UC based on the true solar power
        rd_gscr_result = []
    
    for i in range(load.shape[0]):
        
        # extract the data
        load_true = load[i:i+1]
        solar_true = solar[i:i+1]
        solar_pred = solar_forecast[i:i+1]
        load_pred = load_forecast[i:i+1] if load_forecast is not None else None
        
        # UC performance
        uc_parameters = {
            "load": load_true if load_pred is None else load_pred,  
            "solar": solar_pred
        }
        uc_sol, uc_obj = solve_problem(uc_cvxpy, uc_parameters)
        if W_assessor is not None:
            uc_solar_actual = (solar_pred - uc_sol['solarc']).flatten()
            uc_ug = rd_class.ug_init  # continuous case
            uc_assessor_output = (uc_solar_actual @ W_assessor + b_assessor).flatten()
            if np.any(uc_solar_actual <= 0):
                print('negative solar powe detected: ')
                print('solar pred: ', solar_pred)
                print('solarc: ', uc_sol['solarc'])
                print('solar actual: ', uc_solar_actual)
                exit()
            uc_gscr = small_signal_stability.compute_gSCR(uc_ug, uc_solar_actual) # this is for single sample only
        
        # RD performance
        rd_parameters = {
            "load": load_true,
            "solar": solar_true,
            "pg_parameter": uc_sol['pg']
        }
        rd_sol, rd_obj = solve_problem(rd_cvxpy, rd_parameters)
        if W_assessor is not None:  
            rd_solar_actual = (solar_true - rd_sol['solarc']).flatten()
            rd_ug = rd_class.ug_init
            rd_assessor_output = (rd_solar_actual @ W_assessor + b_assessor).flatten()
            rd_gscr = small_signal_stability.compute_gSCR(rd_ug, rd_solar_actual)
        
        total_cost.append(rd_class.compute_total_cost(uc_sol, rd_sol))
        
        if W_assessor is not None:  
            uc_cls_result.append(uc_assessor_output)
            rd_cls_result.append(rd_assessor_output)
            uc_gscr_result.append(uc_gscr)
            rd_gscr_result.append(rd_gscr)
    
    total_cost = np.array(total_cost)
    print('total_cost: ', np.mean(total_cost))
    performance = {'total_cost': total_cost}
    
    if W_assessor is not None:
        uc_cls = np.array(uc_cls_result)
        rd_cls = np.array(rd_cls_result)
        uc_gscr = np.array(uc_gscr_result)
        rd_gscr = np.array(rd_gscr_result)
        performance.update({
            'uc_cls': uc_cls,
            'rd_cls': rd_cls,
            'uc_gscr': uc_gscr,
            'rd_gscr': rd_gscr
        })
        uc_cls_unstable_ratio = np.sum(uc_cls >= 1e-6) / uc_cls.shape[0]
        rd_cls_unstable_ratio = np.sum(rd_cls >= 1e-6) / rd_cls.shape[0]
        uc_gscr_unstable_ratio = np.sum(uc_gscr <= small_signal_stability.gscr_threshold - 1e-6) / uc_gscr.shape[0]
        rd_gscr_unstable_ratio = np.sum(rd_gscr <= small_signal_stability.gscr_threshold - 1e-6) / rd_gscr.shape[0]
        print(f'UC cls unstable ratio: {uc_cls_unstable_ratio}')
        print(f'RD cls unstable ratio: {rd_cls_unstable_ratio}')
        print(f'UC gSCR unstable ratio: {uc_gscr_unstable_ratio}')
        print(f'RD gSCR unstable ratio: {rd_gscr_unstable_ratio}')
        
    return performance

def add_stability_constraint(cvxpy_prob, W_assessor, b_assessor):
    """
    Add the linear stability constraint to the cvxpy problem
    it is not necessary to transform it into the nn form (as in the sco case), actually
    """
    constraints = cvxpy_prob.constraints
    solar_param = cvxpy_prob.param_dict['solar'] # [T, no_solar]
    solarc_var = cvxpy_prob.var_dict['solarc'] # [T, no_solar]
    constraints.append(((solar_param - solarc_var) @ W_assessor).flatten(order = 'F') + b_assessor <= 0)
    updated_prob = cp.Problem(cvxpy_prob.objective, constraints)
    
    return updated_prob

def subproblem_individual(rd_cvxpy, rd_parameters, load_lower, load_upper, M_RD = 1e4, verbose = True):
    """
    Given the DP solution Pg, solve the worst-case load for the RD for a single sample
    rd_parameters: a dictionary of rd parameters: solar, pg_parameter, load
    
    max_{load} RD objective
    s.t. RD KKT constraints
         load \in [load_lower, load_upper]
    """
    # Convert rd into kkt condition
    kkt_rd_constraints, kkt_rd_variable, rd_param, _ = form_kkt(rd_cvxpy, M_RD)
    
    # decision variable of rd
    rd_var = kkt_rd_variable['x_dict'] # a dictionary of decision variables, go into the objective and links the kkt condition
    rd_var = cp.hstack([rd_var[key] for key in rd_var.keys()]) # flatten the decision variables
    # parameters of rd as variables
    solar_var = kkt_rd_variable['param_dict_as_var']['solar']
    load_var = kkt_rd_variable['param_dict_as_var']['load']
    pg_var = kkt_rd_variable['param_dict_as_var']['pg_parameter']
    
    # Constraints
    constraints = [
        solar_var == rd_parameters['solar'],
        pg_var == rd_parameters['pg_parameter'],
        load_var >= load_lower,
        load_var <= load_upper
    ]
    constraints.extend(kkt_rd_constraints)
    
    # objective
    objective = cp.Maximize(0.5 * cp.quad_form(rd_var, rd_param['P']) + rd_param['q'] @ rd_var)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.GUROBI, verbose = verbose)
    
    # worst uncertainty, part of the upper bound of the UL objective
    return load_var.value, prob.value  # uncertainty is attained at the extreme point

def random_uncertainty(solar_forecast, solar, load, load_list, uc_cvxpy, rd_cvxpy, M_RD, 
                       rd_class
                       ):
    """
    This is used for evaluation
    For a batch of samples to solve the subproblem of the worst-case load for the RD
    start from the forecast and DP
    load_list: (no_data, no_sample, no_load)
    """
    random_total_cost_list = []
    for i in range(load_list.shape[0]):
        random_total_cost = []
        # For each data
        load_true = load[i:i+1]
        load_true_list = load_list[i]
        solar_true = solar[i:i+1]
        solar_pred = solar_forecast[i:i+1]
        
        # UC
        uc_parameters = {"load": load_true,"solar": solar_pred}
        uc_sol, uc_obj = solve_problem(uc_cvxpy, uc_parameters)
        
        # RD
        for j in range(load_true_list.shape[0]):
            # For each sample
            load_random = load_true_list[j:j+1] # [1, no_load]
            # RD
            rd_parameters = {"load": load_random,
                "solar": solar_true,
                "pg_parameter": uc_sol['pg']
            }
            rd_sol, rd_obj = solve_problem(rd_cvxpy, rd_parameters)
            random_total_cost.append(rd_class.compute_total_cost(uc_sol, rd_sol))
        
        random_total_cost_list.append(random_total_cost)
    
    random_total_cost_list = np.array(random_total_cost_list)
    print(f'Random total cost: {np.mean(random_total_cost_list)}')
    return random_total_cost_list
    

def worst_uncertainty(solar_forecast, solar, load, uc_cvxpy, rd_cvxpy, M_RD, 
                      rd_class,
                      budget_ratio = 0.2, 
                      verbose = False):
    """
    This is used for evaluation
    For a batch of samples to solve the subproblem of the worst-case load for the RD
    start from the forecast and DP
    """
    # forecast the solar
    # solar_forecast = feature @ W + b, the optimal solution of the master problem
    
    worst_rd_obj_list = []
    worst_rd_load_list = []
    worst_total_cost_list = []
    for i in range(load.shape[0]):
        
        # Extract the data, T=1
        load_true, solar_true, solar_pred = load[i:i+1], solar[i:i+1], solar_forecast[i:i+1]
        
        # UC
        # TODO: the UC results can be obtained from the master problem
        uc_parameters = {"load": load_true,"solar": solar_pred}
        uc_sol, uc_obj = solve_problem(uc_cvxpy, uc_parameters)
        
        # Worst-case RD
        rd_parameters = {"load": load_true,
            "solar": solar_true,
            "pg_parameter": uc_sol['pg']
        }
        # rd_sol, rd_obj = solve_problem(rd_cvxpy, rd_parameters)
        rd_parameters = {
            "solar": solar_true.flatten(),
            "pg_parameter": uc_sol['pg'].flatten()
        }
        load_lower = np.clip(load_true * (1 - budget_ratio), 0, None)
        load_upper = load_true * (1 + budget_ratio)
        worst_rd_load, worst_rd_obj = subproblem_individual(rd_cvxpy, rd_parameters, load_lower, load_upper,
                                                            M_RD = M_RD,  
                                                            verbose = verbose)
        
        # print('rd_obj: ', rd_obj, 'worst_rd_obj: ', worst_rd_obj)
        
        # Verify the result by solving the RD on worst-case load
        rd_parameters_verify = {
            "load": worst_rd_load[None],
            "solar": solar_true,
            "pg_parameter": uc_sol['pg']
        }
        rd_sol_verify, rd_obj_verify = solve_problem(rd_cvxpy, rd_parameters_verify)
        if np.abs(rd_obj_verify - worst_rd_obj) > 1e-3:
            print('rd_obj: ', rd_obj_verify, 'worst_rd_obj: ', worst_rd_obj)
            print('The result is not correct, potential caused by small M')
            exit()
        # assert np.abs(rd_obj_verify - worst_rd_obj) < 1e-6, 'the result is not correct, potential caused by small M'
    
        worst_total_cost = rd_class.compute_total_cost(uc_sol, rd_sol_verify)
        
        worst_rd_obj_list.append(worst_rd_obj)
        worst_rd_load_list.append(worst_rd_load)
        worst_total_cost_list.append(worst_total_cost)
    
    print(f'Worst-case total cost: {np.mean(worst_total_cost_list)}')
    return np.array(worst_total_cost_list)
    

def solve_main_problem(feature, load, solar, worst_rd_load_list: list,
                    uc_cvxpy, rd_cvxpy, no_gen,
                    solar_min_clip, solar_max,
                    # TODO: Below are not tested
                    Wsolar_acc = None, bsolar_acc = None, alpha = 0.0,
                    M_DP = 1e4,
                    reduced = True, # Only rescale the feature (which is the forecast of the accuracy forecaster)
                    verbose = False):
    """
    Solve the main problem of CCG
    Use the reduced form by moving the lower level RD as upper level constraints
    Only forecast the solar
    main problem constraints: 
    - For all previous iterations and samples,
        - upper bounding the RD objectives
        - RD constraints are satisfied
    - For a single UC (all samples),
        - kkt constraints are satisfied
        - forecast constraints
    """
    # feature: N x no_feature
    # Variables: feature @ Wsolar + bsolar
    if reduced:
        Wsolar_ = cp.Variable(shape = (feature.shape[1],))
        Wsolar = cp.diag(Wsolar_)
    else:
        Wsolar = cp.Variable(shape = (feature.shape[1], solar.shape[1])) # XW + b
    bsolar = cp.Variable(shape = (1, solar.shape[1]))
    
    # UC objective that contributes to the UL objective
    kkt_uc_constraints, kkt_uc_variable, uc_param,_ = form_kkt(uc_cvxpy, M_DP)
    P_uc = uc_param['P'][:no_gen, :no_gen]
    q_uc = uc_param['q'][:no_gen]
    
    # Standard form of RD (becomes the UL constraints)
    P_rd, q_rd, A_rd, G_rd, b_rd, h_rd, B_rd, H_rd = return_standard_form(rd_cvxpy)
    
    # Define variables for RD decision variables
    no_var_rd = P_rd.shape[0]
    no_sample = feature.shape[0]
    no_iteration = len(worst_rd_load_list)
    
    # RD decision variables for each iteration each sample
    z_rd_list = [[cp.Variable(shape = (no_var_rd,)) for _ in range(no_sample)] for _ in range(no_iteration)]
    
    # Upper bound of RD objectives for each sample
    # All iterations should be lower than this value
    eta = cp.Variable(shape = (no_sample,))
    
    # Constraints
    constraints = [eta >= 0] # eta is non-negative (essential for first iteration)
    first_stage_obj = 0
    
    # Forecast constraints in UL
    solar_forecast_all = feature @ Wsolar + bsolar  # XW + b [N, no_solar]
    constraints += [solar_forecast_all >= solar_min_clip, solar_forecast_all <= solar_max[None]]
        
    # # Dummy constraints to make the model feasible
    # constraints += [Wsolar == np.eye(solar.shape[1])]
    # constraints += [bsolar == np.zeros((1, solar.shape[1]))]
    
    pg_uc_list = [] # record the pg of UC for each sample, which will be sent to the sub problem
    
    for i in range(no_sample):
        
        # LL UC constraints
        solar_forecast = solar_forecast_all[i] # [no_solar] pass to the UC
        load_true = load[i] # pass to the UC
        
        # regenerate the UC kkt constraints again as new variables need to be built
        kkt_uc_constraints, kkt_uc_variable, uc_param,_ = form_kkt(uc_cvxpy, M_DP) 
        param_uc_var = kkt_uc_variable['param_dict_as_var']
        z_uc = kkt_uc_variable['x_dict']
        
        constraints.extend(kkt_uc_constraints)   # UC KKT constraints
        constraints+= [ 
            # assign the forecast to the parameters (actually another variable)
            param_uc_var['load'] == load_true,
            param_uc_var['solar'] == solar_forecast,
            ]
        
        # UL RD constraints
        solar_true = solar[i] # pass to the RD
        
        # Optimality Cut
        for j in range(no_iteration):
            # For each iteration
            constraints+= [
                # rd constraints becomes the UL constraints
                A_rd @ z_rd_list[j][i] == b_rd + B_rd['load'] @ worst_rd_load_list[j][i] \
                    + B_rd['solar'] @ solar_true + B_rd['pg_parameter'] @ z_uc['pg'],
                G_rd @ z_rd_list[j][i] <= h_rd + H_rd['load'] @ worst_rd_load_list[j][i] \
                    + H_rd['solar'] @ solar_true + H_rd['pg_parameter'] @ z_uc['pg'],
                # lower bound of RD objective
                eta[i] >= 0.5 * cp.quad_form(z_rd_list[j][i], P_rd) + q_rd @ z_rd_list[j][i]
            ]
        
        pg_uc_list.append(z_uc['pg'])
        
        # Objective
        first_stage_obj += 0.5 * cp.quad_form(z_uc['pg'], P_uc) + q_uc @ z_uc['pg']
        
    # Add the RD upper bound
    first_stage_obj = first_stage_obj/no_sample
    second_stage_obj = cp.sum(eta)/no_sample
    obj = first_stage_obj + second_stage_obj
    
    # Add the regularization term
    # NOTE: Not in function yet
    assert alpha == 0, 'Regularization term is not tested'
    if alpha > 0:
        obj += alpha * (
            cp.mean(cp.abs(Wsolar - Wsolar_acc)) + cp.mean(cp.abs(bsolar - bsolar_acc)))
        
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver = cp.GUROBI, verbose = verbose)
    
    pg_uc_list = np.array([pg_uc_value.value for pg_uc_value in pg_uc_list])
    
    return Wsolar.value, bsolar.value, pg_uc_list, first_stage_obj.value, second_stage_obj.value, obj.value

def ccg(feature, load, solar,
        uc_cvxpy, rd_cvxpy, no_gen,
        solar_min_clip, solar_max,
        budget_ratio,
        reduced = True,
        # TODO: Below are not tested
        Wsolar_acc = None, bsolar_acc = None, alpha = 0.0,
        M_DP = 1e4, M_RD = 1e4, max_iter = 100, tol = 1e-3, verbose = False):
    
    """
    use ccg to solve two-stage robust optimization problem
    """
    # Initialization
    LB, UB = -1e6, 1e6
    worst_rd_load_list = [] # Iteratively updated worst-case load in RD
    for iter in range(max_iter):
        Wsolar, bsolar, pg_uc_list, obj_first_stage, obj_second_stage, main_obj = solve_main_problem(
                                                            feature, load, solar, 
                                                            worst_rd_load_list,
                                                            uc_cvxpy, rd_cvxpy, no_gen,
                                                            solar_min_clip, solar_max,
                                                            Wsolar_acc, bsolar_acc, alpha,
                                                            M_DP,
                                                            reduced = reduced,
                                                            verbose = verbose)
        
        # update the lower bound
        LB = main_obj
        # solve the subproblem for each sample: can be done in parallel if each one is complex to solve
        worst_load_list, worst_obj_list = [], []
        # We did not call the previous function, because we can reuse the DP solution
        for i in range(feature.shape[0]):
            load_lower = np.clip(load[i] * (1 - budget_ratio), 0, None)
            load_upper = np.clip(load[i] * (1 + budget_ratio), 0, None)
            
            solar_true = solar[i]
            pg_uc = pg_uc_list[i]
            worst_load, worst_obj = subproblem_individual(rd_cvxpy, {
                'solar': solar_true,
                'pg_parameter': pg_uc
            }, load_lower, load_upper, M_RD, False)
            worst_load_list.append(worst_load)
            worst_obj_list.append(worst_obj)
        
        worst_rd_load_list.append(worst_load_list)  # update the iterative worst-case load
        
        Q = np.sum(worst_obj_list) / feature.shape[0] + obj_first_stage
        UB = min(UB, Q)
        print(f'Iteration {iter}: UB: {UB}, LB: {LB}')
        
        if UB - LB < tol:
            print(f'Converged at iteration {iter}')
            break
    
    return Wsolar, bsolar, UB

def return_grad_acc(W, b, feature, solar):
    """
    Find the gradient of accuracy training objective with respect to the forecast weight (and bias)
    """
    
    # print("W shape: ", W.shape)
    # print("b shape: ", b.shape)
    # print("feature shape: ", feature.shape)
    # print("solar shape: ", solar.shape)
    
    solar_true = torch.from_numpy(solar).float()
    feature = torch.from_numpy(feature).float()
    
    grad_W_list = []
    grad_b_list = []
    
    W = torch.from_numpy(W).float().requires_grad_(True)
    b = torch.from_numpy(b).float().requires_grad_(True)
    
    for i in range(feature.shape[0]):
        forecast = feature[i:i+1] @ W + b
        error = torch.abs(forecast - solar_true[i:i+1])
        loss = torch.mean(error)
        loss.backward()
        
        grad_W = W.grad.flatten()
        grad_b = b.grad.flatten()
        
        W.grad.zero_()
        b.grad.zero_()
        
        grad_W_list.append(grad_W.detach().numpy()) 
        grad_b_list.append(grad_b.detach().numpy())
    
    return np.array(grad_W_list), np.array(grad_b_list)
        
    
def return_grad(W, b, feature, load, solar, uc_cvxpy, rd_cvxpy, rd_class):
    """
    Find the gradient of the total cost with respect to the forecast weight (and bias)
    """        
    
    load_true = torch.from_numpy(load).float()[:,None,:] # add the T dimension
    solar_true = torch.from_numpy(solar).float()[:,None,:] # add the T dimension
    feature = torch.from_numpy(feature).float()
    
    first_coeff = torch.from_numpy(rd_class.first).float()
    cls_coeff = torch.from_numpy(rd_class.cls).float()
    storage_coeff = torch.from_numpy(rd_class.storage).float()
    solarc_coeff = torch.from_numpy(rd_class.csc).float()
    
    class ForecastOpt(torch.nn.Module):
        def __init__(self, no_feature, no_solar, uc_cvxpy, rd_cvxpy, W, b):
            super(ForecastOpt, self).__init__()
            self.linear = torch.nn.Linear(no_feature, no_solar, bias=True)
            self.uc_layer = CvxpyLayer(uc_cvxpy, parameters=uc_cvxpy.parameters(), variables=uc_cvxpy.variables())
            self.rd_layer = CvxpyLayer(rd_cvxpy, parameters=rd_cvxpy.parameters(), variables=rd_cvxpy.variables())
            self.linear.weight.data = torch.from_numpy(W).float().T # Initialize the weight with the transpose of W
            self.linear.bias.data = torch.from_numpy(b).float()
            
        def forward(self, x, load_true, solar_true):
            forecast = self.linear(x)
            uc_solution = self.uc_layer(load_true, forecast, solver_args={"solve_method": "ECOS"})
            rd_solution = self.rd_layer(uc_solution[0], load_true, solar_true, solver_args={"solve_method": "ECOS"})
            return forecast, uc_solution, rd_solution
    
    # Construct an NN model with linear layer as values in W and b
    net = ForecastOpt(feature.shape[1], solar.shape[1], uc_cvxpy, rd_cvxpy, W, b)
    
    total_cost = []
    grad_W_list = []
    grad_b_list = []
    
    success_idx = []
    for i in range(len(solar_true)):
        
        try:
            forecast, uc_solution, rd_solution = net(feature[i:i+1], load_true[i:i+1], solar_true[i:i+1])
                    
            # total cost
            pg_uc = uc_solution[0].flatten()
            delta_pg = rd_solution[0].flatten()
            ls = rd_solution[1].flatten()
            rd_cost = rd_solution[2].flatten()
            es = rd_solution[3].flatten()
            solarc = rd_solution[4].flatten()
            
            cost = torch.inner(first_coeff, pg_uc + delta_pg) + torch.inner(cls_coeff, ls) + torch.inner(storage_coeff, es) + torch.inner(solarc_coeff, solarc) + rd_cost.sum()
            cost.backward()
            
            grad_W = net.linear.weight.grad.flatten()
            grad_b = net.linear.bias.grad.flatten()
            total_cost.append(cost.item())
            grad_W_list.append(grad_W.detach().numpy())
            grad_b_list.append(grad_b.detach().numpy())
            net.zero_grad()
            success_idx.append(i)
        except Exception as e:
            print(f'Error at sample {i}: {e}')
            total_cost.append(np.nan)
            grad_W_list.append(np.zeros_like(grad_W))
            grad_b_list.append(np.zeros_like(grad_b))
            continue
        
    return np.array(total_cost), np.array(grad_W_list), np.array(grad_b_list), success_idx
        
        
        
        
# def evaluate(Wsolar, bsolar, load, solar, uc, rd, feature = None):
#     """
#     only forecast the solar
#     the true load is passed as parameter
#     feature: N x no_feature
#     solar_forecast_acc: N x no_solar
#     """
    
#     # Accuracy
#     print('Regression model norms:')
#     print('Wsolar: ', np.linalg.norm(Wsolar), 'bsolar: ', np.linalg.norm(bsolar))
#     print(f'Wsolar max: {np.max(np.abs(Wsolar))}')
#     print(f'bsolar max: {np.max(np.abs(bsolar))}')
    
#     # accuracy model
#     solar_pred = feature @ Wsolar + bsolar
#     solar_mape = np.mean(np.abs(solar_pred - solar) / solar)
#     solar_rmse = np.sqrt(np.mean(np.square(solar_pred - solar)))
#     print(f'Solar MAPE: {solar_mape}')
#     print(f'Solar RMSE: {solar_rmse}')
    
#     solar_diff = solar_pred.sum(axis=1) - solar.sum(axis=1)
#     print('solar diff (positive means over-forecast): ', np.mean(solar_diff), np.std(solar_diff))
    
#     # Cost
#     total_cost = []
#     total_true_cost = []
    
#     for i in range(load.shape[0]):
        
#         load_true_ = load[i:i+1]
#         solar_true_ = solar[i:i+1]
#         solar_pred_ = np.clip(solar_pred[i:i+1], 0, None)
        
#         uc_parameters = {
#             "load": load_true_,
#             "solar": solar_pred_
#         }
#         uc_result = uc.solve(uc_parameters)
        
#         rd_parameters = {
#             "load": load_true_,
#             "solar": solar_true_,
#             "pg_parameter": uc_result['pg']
#         }
#         rd_result = rd.solve(rd_parameters)
#         total_cost.append(rd.compute_total_cost(uc_result, rd_result))
        
#         uc_parameters_true = {
#             "load": load_true_,
#             "solar": solar_true_
#         }
#         uc_result_true = uc.solve(uc_parameters_true)
#         rd_parameters_true = {
#             "load": load_true_,
#             "solar": solar_true_,
#             "pg_parameter": uc_result_true['pg']
#         }
#         rd_result_true = rd.solve(rd_parameters_true)
#         total_true_cost.append(rd.compute_total_cost(uc_result_true, rd_result_true))
        
#     print(f'Total cost: {np.mean(np.array(total_cost))}')
#     print(f'Total true cost: {np.mean(np.array(total_true_cost))}')
    
#     return solar_pred, total_cost, total_true_cost

