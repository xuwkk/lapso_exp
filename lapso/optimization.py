"""
return the standard form of the operation problem as QP or LP
"""

import cvxpy as cp
import numpy as np

def return_compiler(prob):
    """
    return the compiler of the problem given by cvxpy
    return:
        - compiler: the compiler of the problem in standard form
        - params_idx: {param_id: param_name}, link the id to the parameter name
        - zero_dim: the number of equality constraint in the standard form 
                                        (always be the first rows in the matrix)
        - int_vars_idx: the index of integer variables
        - bool_vars_idx: the index of boolean variables
    """

    data, _, _ = prob.get_problem_data(
                solver=cp.GUROBI, solver_opts={'use_quad_obj': True})  
    # ! set True can force the objective to be quadrtic

    assert prob.is_qp(), 'only support QP (and LP) for now'
    assert data['dims'].exp == 0, 'does not support exponential cone'
    assert len(data['dims'].psd) == 0, 'does not support positive semidefinite cone'
    assert len(data['dims'].soc) == 0, 'does not support second-order cone'

    # parametric QP problem
    param_qp_prog = data[cp.settings.PARAM_PROB]
    
    # ! the order of parameter idx is changed internally in cvxpy so we link the id to the name
    params_idx = {p.id: p.name() for p in prob.parameters()}  

    return param_qp_prog, params_idx, data['dims'].zero, data['int_vars_idx'], data['bool_vars_idx']

def return_standard_form_with_value(prob, params_val_dict):
    """find the compiler first to save time
    prob: a cvxpy problem
    return standard form as
    min 1/2 x^T P x + q^T x
    s.t. A x = b
         G x <= h
    in which x is the decision variable
    the order of params_val should be the same as the param_ids
    output[0]: P (with 1/2 being considered)
    output[1]: q
    output[2]: r
    output[3]: [eq_matrix; ineq_matrix]
    output[4]: [eq_vec; ineq_vec]"""

    param_qp_prog, params_idx, zero_dim, int_dim, bool_dim = return_compiler(prob)
    params_val = {idx: params_val_dict[name] for idx, name in params_idx.items()}
    
    output = param_qp_prog.apply_parameters(
                    params_val,
                    keep_zeros=True)
    
    P = output[0].toarray()
    q = output[1]
    r = output[2]
    A = output[3].toarray()[:zero_dim]
    b = -output[4][:zero_dim]
    G = -output[3].toarray()[zero_dim:]
    h = output[4][zero_dim:]
    
    return P, q, r, A, b, G, h

def return_bool_idx(prob):
    """
    return the index of the boolean variables
    prob: a cvxpy problem
    """
    data, _, _ = prob.get_problem_data(
                solver=cp.GUROBI, solver_opts={'use_quad_obj': True})  
    
    return data['bool_vars_idx']

def return_standard_form(prob): 
    """
    standard form of the QP problem without parameter value
    idx_to_name: {param_id: param_name}, link the id to the parameter name
    
    The standard form is:
    min 1/2 x^T P x + q^T x
    s.t. A x = b + \sum B_i z_i
         G x <= h + \sum H_i z_i
    in which x is the decision variable, z_i is the i-th parameter
    """
    # Creates a dictionary mapping parameter IDs to their names
    param_id_to_name = {p.id: p.name() for p in prob.parameters()}

    # Compiles the QP problem and returns:
    # - param_qp_prog: the compiled QP program
    # - params_idx: {param_id: param_name}, link the id to the parameter name
    # - zero_dim: number of equality constraints
    # - int_dim: number of integer constraints
    # - bool_dim: number of boolean constraints
    param_qp_prog, params_idx, zero_dim, int_dim, bool_dim = return_compiler(prob)

    # Get the total number of constraints and variables
    no_cons = param_qp_prog.constr_size
    no_var = param_qp_prog.reduced_A.var_len

    # Extract the quadratic cost matrix P
    # toarray() converts sparse matrix to dense
    # reshape to square matrix of size no_var × no_var
    P = param_qp_prog.P.toarray()[:,-1].reshape(no_var, no_var)
    
    # Extract the linear cost vector q
    q = param_qp_prog.q.toarray()[:-1,-1]

    # Extract and reshape the constraint matrix A_tilde
    # This contains both equality and inequality constraints
    A_tilde = param_qp_prog.A.toarray()[:int(no_cons * no_var), -1]
    A_tilde = A_tilde.reshape(no_var, no_cons).T

    # Extract the right-hand side vector b_tilde
    b_tilde = param_qp_prog.A.toarray()[int(no_cons * no_var):, -1]

    # Split A_tilde into:
    # A: equality constraints (first zero_dim rows)
    # G: inequality constraints (remaining rows, with negative sign)
    A = A_tilde[:zero_dim,:]
    G = -A_tilde[zero_dim:,:]

    # Split b_tilde into:
    # b: equality constraints right-hand side (with negative sign)
    # h: inequality constraints right-hand side
    b = -b_tilde[:zero_dim]
    h = b_tilde[zero_dim:]

    # Extract the parameter coefficient matrix B_tilde
    B_tilde = param_qp_prog.A.toarray()[int(no_cons * no_var):, :-1]

    # Initialize dictionaries for parameter matrices
    B = {}  # For equality constraints
    H = {}  # For inequality constraints

    # Get mappings for parameters
    param_id_to_col = param_qp_prog.param_id_to_col    # Maps parameter ID to starting column
    param_id_to_size = param_qp_prog.param_id_to_size  # Maps parameter ID to its size

    # For each parameter:
    for key, start_idx in param_id_to_col.items():
        if key == -1:  # Skip if key is -1 (special value)
            break
        
        size = param_id_to_size[key]        # Get parameter size
        name = param_id_to_name[key]        # Get parameter name
        
        # Extract parameter matrices:
        # B[name]: how parameter affects equality constraints
        # H[name]: how parameter affects inequality constraints
        B[name] = -B_tilde[:zero_dim, start_idx:start_idx+size]
        H[name] = B_tilde[zero_dim:, start_idx:start_idx+size]

    return P, q, A, G, b, h, B, H # NOTE: negative sign

def return_standard_form_in_cvxpy(prob):
    """
    return the standard form of the problem fommated as cvxpy
    standard form
    min 1/2 x^T P x + q^T x
    s.t. A x = b + \sum B_i z_i
         G x <= h + \sum H_i z_i
    in which x is the decision variable, z_i is the i-th parameter
    NOTE: the formulation accepts multiple parameters for both inequality and equality constraints
    NOTE: you still need to assign the parameters to the problem after this conversion
    """
    
    P, q, A, G, b, h, B, H = return_standard_form(prob)
    bool_idx = return_bool_idx(prob)
    
    x = cp.Variable(P.shape[1])
    parameters = {
        key: 
        cp.Parameter(B[key].shape[1], name = key) for key in B.keys()
        } # paramters for the standard QP
    
    # formulate the cvxpy problem
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)
    constraints = []
    if len(bool_idx) > 0:
        # set the integer (binary) constraints
        constraints += [cp.FiniteSet(x[bool_idx], [0, 1])]
    
    b_ = 0
    h_ = 0
    
    for key in B.keys():
        b_ += B[key] @ parameters[key]
        h_ += H[key] @ parameters[key]
    
    constraints += [A @ x == b_ + b, G @ x <= h_ + h]
    
    prob = cp.Problem(objective, constraints)
    
    return prob, (P, q, A, G, b, h, B, H)

def return_kkt_result(P,q,A,G,b,h,B,H,x,ineq_multiplier,eq_multiplier,param_dict):
    """
    return the result of the KKT conditions
    """
    b_ = 0
    h_ = 0
    
    for key in B.keys():
        b_ += B[key] @ param_dict[key]
        h_ += H[key] @ param_dict[key]
    
    return P @ x + q + A.T @ eq_multiplier + G.T @ ineq_multiplier, A@x-b-b_, G@x-h-h_,ineq_multiplier * (G@x-h-h_)

def form_kkt(prob, M):
    """
    Args:
        prob: the cvxpy problem with parameters name and variable name clearly defined
        M: the large constant provided by the user for forming big-M
    Form the KKT conditions as constraint of cvxpy problem
    Returns:
        kkt_constraints: the constraints of the KKT conditions
        kkt_variable: {'x': dictionary of original variables, 
                        'ineq_multiplier': inequality multiplier in kkt, 
                        'eq_multiplier': equality multiplier in kkt, 
                        'phi': parameter for kkt binary variables, 
                        'param_dict_as_var': dictionary of parameters as variables}
        standard_form_matrix: {'P': P, 'q': q, 'A': A, 'G': G, 'b': b, 'h': h, 'B': B, 'H': H}
    
    Given convex (only support LP and QP for now)
    min 1/2 x^T P x + q^T x
    s.t. A x = b + \sum B_i z_i
         G x <= h + \sum H_i z_i
    in which x is the decision variable, z_i is the i-th parameter
    
    NOTE: the parameters are considered as variables
    
    The KKT conditions include:
    Px + q + A^T \mu + G^T lambda = 0
    A x = b + \sum B_i z_i
    G x <= h + \sum H_i z_i
    diag(lambda) (Gx - h - \sum H_i z_i) = 0
    lambda >= 0
    which can be written into more compact form:
    Px + q + A^T \lambda + G^T \mu = 0
    Ax = b + \sum B_i z_i
    0 \leq lambda \perp Gx - h - \sum H_i z_i \geq 0
    which can be rewritten by complementarity linearization:
    Px + q + A^T \lambda + G^T \mu = 0
    Ax = b + \sum B_i z_i
    lambda \geq 0, Gx - h - \sum H_i z_i \geq 0, lambda <= phi M, Gx - h - \sum H_i z_i <= (1-phi) M
    where M is a large constant provided by the user
    """
    
    var_dict = {}
    for item in prob.variables():
        var_dict[item.name()] = item.shape
    
    P, q, A, G, b, h, B, H = return_standard_form(prob)
    
    # formualte the KKT conditions as constraints
    # we can consider the parameters as another variable
    
    x_dict = {}
    for key in var_dict.keys():
        x_dict[key] = cp.Variable(np.prod(var_dict[key]), name = key)  # original shape
    
    x = cp.hstack([x_dict[key] for key in x_dict.keys()]) # flatten and concatenate the variables
    
    # x = cp.Variable(P.shape[1]) # the original decision variable
    param_dict_as_var = {key: cp.Variable(B[key].shape[1], name = key) for key in B.keys()} # the parameters as variables
    ineq_multiplier = cp.Variable(G.shape[0], nonneg = True) # the multiplier for the inequality constraints
    eq_multiplier = cp.Variable(A.shape[0]) # the multiplier for the equality constraints
    phi = cp.Variable(G.shape[0], boolean = True) # the parameter for the complementarity linearization
    
    # the KKT conditions
    b_ = 0
    h_ = 0
    for key in B.keys():
        b_ += B[key] @ param_dict_as_var[key]
        h_ += H[key] @ param_dict_as_var[key]
    constraints = [
        # stationary condition
        P @ x + q + A.T @ eq_multiplier + G.T @ ineq_multiplier == 0,
        # equality constraints
        A @ x == b + b_,
        # inequality constraints via complementarity linearization
        # ineq_multiplier >= 0,
        G@x - h - h_ <= 0,
        ineq_multiplier <= phi * M,
        G @ x - h - h_ >= (phi - 1) * M
    ]
    
    # return constraints, (x, var_dict, param_dict_as_var, 
    #                     ineq_multiplier, eq_multiplier, phi), (P, q, A, G, b, h, B, H)
    
    # todo: change to better store the shape of the variables
    return constraints, {'x_dict': x_dict, 'ineq_multiplier': ineq_multiplier, 'eq_multiplier': eq_multiplier, 'phi': phi, 
                        'param_dict_as_var': param_dict_as_var}, {'P': P, 'q': q, 'A': A, 'G': G, 'b': b, 'h': h, 'B': B, 'H': H}, var_dict