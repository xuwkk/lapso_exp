"""
Helper function for the SCO experiment
"""
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from omegaconf import open_dict

class SmallSignalStability:
    def __init__(self, uc, xd, gscr_threshold, solar_min_clip, solar_max):
        # Construct reordered Bbus matrix in the order [renewable, generator, other]
        # generator bus is also listed as the reactance needs to be added

        # Sort by idx type
        self.solar_idx = uc.solar_idx
        self.gen_idx = uc.gen_idx  # start from zero

        self.not_gen_ren_idx = uc.not_gen_ren_idx  # The remaining (other) bus indices
        self.no_bus = uc.no_bus
        self.no_solar = uc.no_solar
        self.no_gen = uc.no_gen

        # If a row idx is to be moved to row i, then new_identity[idx, i]^T = 1
        # If a column idx is to be moved to column i, then new_identity[idx, i] = 1
        # Therefore, Bbus_sorted = new_identity.T @ self.Bbus @ new_identity
        new_identity = np.zeros((uc.no_bus, uc.no_bus))
        for i, idx in enumerate(uc.solar_idx):
            new_identity[idx, i] = 1
        start_idx = uc.no_solar
        
        # ! in case there is a bus contains more than one generators
        checked_gen_idx = []
        i = 0
        for idx in uc.gen_idx:
            # [1,1,2,3,6,8]
            # move row [1,2,3,6,8] to start_idx + [1,2,3,4,5]
            if not idx in checked_gen_idx:
                new_identity[idx, i + start_idx] = 1
                i += 1
            checked_gen_idx.append(idx)
            
        start_idx += len(checked_gen_idx)

        for i, idx in enumerate(uc.not_gen_ren_idx):
            new_identity[idx, i + start_idx] = 1
        
        Bbus_sorted = new_identity.T @ uc.Bbus @ new_identity

        # test
        assert np.allclose(Bbus_sorted, Bbus_sorted.T), "The sorted Bbus is not symmetric."
        assert np.allclose(
                    uc.Bbus[uc.solar_idx,:][:, uc.solar_idx], 
                    Bbus_sorted[:uc.no_solar, :uc.no_solar]
                    ), "The solar part is not correct."
        assert np.allclose(
                    uc.Bbus[uc.gen_idx,:][:, uc.gen_idx], 
                    Bbus_sorted[uc.no_solar:uc.no_solar + len(checked_gen_idx), 
                                uc.no_solar:uc.no_solar + len(checked_gen_idx)]
                    ), "The generator part is not correct."
        
        gen_start_idx = uc.no_solar + uc.no_wind
            
        # gen_bus_idx: [1,2,3,6,8] move these rows after renewables
        # gen_idx: [1,1,2,3,6,8]
        assert np.allclose(
            uc.Bbus[checked_gen_idx,:][:, checked_gen_idx], 
            Bbus_sorted[gen_start_idx:gen_start_idx + len(checked_gen_idx), gen_start_idx:gen_start_idx + len(checked_gen_idx)]
            ), "The generator part is not correct."
            
        non_gen_ren_start_idx = uc.no_solar + uc.no_wind + len(checked_gen_idx)
        assert np.allclose(
            uc.Bbus[uc.not_gen_ren_idx,:][:, uc.not_gen_ren_idx], 
            Bbus_sorted[non_gen_ren_start_idx:, non_gen_ren_start_idx:]
            ), "The not generator and renewable part is not correct."
        
        # Bbus_sorted is a (no_bus, no_bus) symmetric matrix in (solar, gen, remaining) order
        self.Bbus_sorted = Bbus_sorted   
        self.Cg_to_gen_bus = uc.Cg_to_gen_bus # incidence matrix of generator to bus with generator (including multiple generators on the same bus)
        self.xd = xd  # the reactance of the generator
        self.gscr_threshold = gscr_threshold  # the threshold of the generalized short-circuit ratio
        self.solar_min_clip = solar_min_clip  # the minimum clip of the solar power > 0
        self.solar_max = solar_max  # the maximum of the solar power

    # small-signal stability analysis
    def compute_gSCR(self, ug, psolar):
        """
        Compute the generalized short-circuit ratio as the small-signal stability metric
        return the input feature and label
        """
        # IBR output
        
        assert np.sum(ug) >= 1, "At least one generator is on."
        # TODO: should allow zero solar power but this will make the Bbus_sorted changes
        assert np.all(psolar > 0), "All solar power is positive."

        vren = np.ones(self.no_solar) # by assumption, the IBR terminal voltage during transient is 1.0 p.u.
        
        # Admittance matrix of generator
        # ! check this
        Bg = np.zeros(self.no_bus)
        # generator contribution to the Bbus, allow more than one generator on the same bus
        Bg[self.no_solar:self.no_solar + self.no_gen] = self.Cg_to_gen_bus.dot(ug / self.xd)  # this will sum up multiple generators xd on the same bus
        Bg = np.diag(Bg)
        
        # kron reduction: reveal the renewable part
        B = self.Bbus_sorted + Bg    # Generator contributes some reactance when fault occurs    
        B11 = B[:self.no_solar, :self.no_solar]
        B12 = B[:self.no_solar, self.no_solar:]
        B21 = B[self.no_solar:, :self.no_solar]
        B22 = B[self.no_solar:, self.no_solar:]
        
        Bred = B11 - B12 @ np.linalg.inv(B22) @ B21  # This can be slow but as B22 is dependent on ug, we have to do it case-by-case
        
        assert np.allclose(Bred, Bred.T), "The reduced Bbus is not symmetric."
        
        ## generalized short circuit ratio
        try:
            Beq = np.diag(vren**2 / psolar) @ Bred
            eig = np.linalg.eigvals(Beq)
            gSCR = np.min(eig)
            if gSCR < 0:
                print("Negative gSCR found!")
                print('gSCR is negative:', gSCR)
                print('psolar:', psolar)
                print('ug:', ug)
                exit()
        except:
            print('psolar:', psolar)
            print('ug:', ug)
            exit()
        
        return gSCR
    
    def compute_gscr_torch(self, ug, psolar):
        """
        Use torch to compute the gSCR
        Differentiation with respect to psolar and ug
        ug and psolar are tensors
        """

        dtype = torch.float32
        # Convert constant to tensor
        xd = torch.tensor(self.xd, dtype = dtype)
        Cg_to_gen_bus = torch.tensor(self.Cg_to_gen_bus, dtype = dtype)
        Bbus_sorted = torch.tensor(self.Bbus_sorted, dtype = dtype)

        vren = torch.ones(self.no_solar, dtype = dtype) # by assumption, the IBR terminal voltage during transient is 1.0 p.u.

        Bg_vec = torch.zeros(self.no_bus, dtype = dtype)
        gen_sus = ug / xd  # differentiable w.r.t. ug
        mapped = Cg_to_gen_bus @ gen_sus  # shape must match the generator-bus slice length

        # place into the generator-bus portion of Bg
        Bg_vec[self.no_solar:self.no_solar + self.no_gen] = mapped

        # Form Bg diagonal and total B
        Bg = torch.diag(Bg_vec)
        B = Bbus_sorted + Bg

        # Partition B
        B11 = B[:self.no_solar, :self.no_solar]
        B12 = B[:self.no_solar, self.no_solar:]
        B21 = B[self.no_solar:, :self.no_solar]
        B22 = B[self.no_solar:, self.no_solar:]

        X = torch.linalg.solve(B22, B21)               # solves for X
        Bred = B11 - B12 @ X

        # Symmetric similarity to ensure real eigenvalues:
        # S = diag(1/sqrt(psolar)), Beq_sym = S @ Bred @ S
        S = torch.diag(torch.rsqrt(psolar))
        Beq_sym = S @ Bred @ S

        # Ensure symmetry numerically (tiny asymmetry can creep in)
        Beq_sym = 0.5 * (Beq_sym + Beq_sym.transpose(-1, -2))

        # Smallest eigenvalue (real, differentiable)
        # eigvalsh is for symmetric/hermitian matrices
        eigvals = torch.linalg.eigvalsh(Beq_sym)
        gscr = eigvals.min()

        return gscr

    # dataset generation
    def gen_dataset_small_signal(self, solar_sample_no, solar_sample_ratio_list=None, verbose = False, 
                                fixed_ug = None):
        """
        Construct the dataset for the small-signal stability analysis.
        gSCR is a function of ug and psolar.
        Args:
            - solar_sample_no: the number of evenly-spaced samples between solar_min_clip and solar_max
            - fixed_ug: the fixed generator status, if None, then sample all the possible generator status
        """
        
        # Go through all the 2^no_gen possible generator status
        ug_samples = []
        if fixed_ug is None:
            for i in range(2**self.no_gen):
                ug_samples.append([int(x) for x in list(bin(i)[2:].zfill(self.no_gen))])
            # print('ug sample number:', len(ug_samples))
            # remove the all-off status
            ug_samples = ug_samples[1:]
        
        # sample the renewable power uniformly
        solar_samples = []
        
        if solar_sample_ratio_list is None:
            solar_step_ratio = np.linspace(0, 1, solar_sample_no)
        else:
            solar_step_ratio = solar_sample_ratio_list
        for i in range(self.no_solar):
            solar_samples_ = self.solar_min_clip + (self.solar_max[i] - self.solar_min_clip) * solar_step_ratio
            solar_samples.append(solar_samples_)
        
        solar_samples = np.meshgrid(*solar_samples) # return the full coordinate matrix
        solar_samples = [x.flatten() for x in solar_samples]
        
        solar_samples = np.array(solar_samples) # (no_ren, no_samples**no_ren)
        ## construct the input
        input_space = []
        if fixed_ug is None:
            # Combine the generator status and the solar power
            for ug in ug_samples:
                for i in range(solar_samples.shape[1]):                
                    input_space.append(
                        np.concatenate([ug, solar_samples[:, i]])
                    )
        else:
            # Only with solar samples
            input_space = solar_samples.T
        
        input_space = np.array(input_space)
        
        # Check if all the entry of input_space is unique
        assert len(np.unique(input_space, axis = 0)) == input_space.shape[0], "The input space is not unique."
        
        ## Construct the output by evaluating the gSCR on the input space
        output_space = []
        for i in range(input_space.shape[0]):
            if fixed_ug is None:
                psolar = input_space[i, self.no_gen:self.no_gen + self.no_solar]
                ug = input_space[i, :self.no_gen]
            else:
                # constant ug
                psolar = input_space[i]
                ug = fixed_ug
            
            output_space.append(self.compute_gSCR(ug, psolar))
        
        if verbose:
            print("The input space size: ", input_space.shape)
            print("The output space size: ", len(output_space))
            print("Some small signal stability examples (more generator and less renewable, the larger the gSCR):")
            print("Example 1: ", np.round(input_space[0],2), np.round(output_space[0],2))
            print("Example 2: ", np.round(input_space[len(output_space) // 2],2), np.round(output_space[len(output_space) // 2],2))
            print("Example 3: ", np.round(input_space[-1], 2), np.round(output_space[-1], 2))
        
        return input_space, np.array(output_space)
    
    def perturb_solar(self, ug_numpy, psolar_numpy, lr, max_iter = 20):
        """
        Perturb the solar power to change the gscr from stable to unstable or vice versa
        """
        gscr_ori = self.compute_gSCR(ug_numpy, psolar_numpy)

        ug = torch.tensor(ug_numpy, dtype=torch.float32, requires_grad=False)
        psolar = torch.tensor(psolar_numpy, dtype=torch.float32, requires_grad=True)

        if gscr_ori >= self.gscr_threshold:
            lr = -lr # from stable to unstable

        psolar_prev = psolar.clone().detach().numpy()
        
        for i in range(max_iter):
            
            with torch.enable_grad():
                gscr = self.compute_gscr_torch(ug, psolar)
            # print(f"Iteration {i}, gSCR: {gscr.item()}, psolar: {psolar.detach().numpy()}")
            # Termination flag
            if lr * (gscr.item() - self.gscr_threshold) > 0:
                # change sign
                break

            psolar_prev = psolar.clone().detach().numpy()

            gscr.backward()
            grad_psolar = psolar.grad
                
            psolar.data = torch.clamp(psolar + lr * grad_psolar / torch.norm(grad_psolar), # Normalized gradient step 
                                    min = torch.tensor(self.solar_min_clip), 
                                    max = torch.tensor(self.solar_max, dtype=torch.float32))
            psolar.grad.zero_()
        
        return psolar.detach().numpy(), psolar_prev, i
    
    def perturb_ug(self, ug_numpy, psolar_numpy, on_cost):
        """
        Perturb the generator status to change the gscr from stable to unstable or vice versa
        on_cost: the cost of turning on each generator
        """
        cost_ascending_idx = np.argsort(on_cost) # ascending order
        gscr_ori = self.compute_gSCR(ug_numpy, psolar_numpy)
        ug = deepcopy(ug_numpy)

        ug_prev = deepcopy(ug)
        for i in range(self.no_gen):
            if gscr_ori >= self.gscr_threshold and np.sum(ug) > 1:
                # stable to unstable, turn off the most expensive generator
                ug_prev = deepcopy(ug)
                for i in cost_ascending_idx[::-1]:
                    if ug[i] > 0:
                        ug[i] = 0
                        break
            elif gscr_ori < self.gscr_threshold and np.sum(ug) < self.no_gen:
                # unstable to stable, turn on the cheapest generator
                ug_prev = deepcopy(ug)
                for i in cost_ascending_idx:
                    if ug[i] < 1:
                        ug[i] = 1
                        break
            else:
                break
            
            gscr_new = self.compute_gSCR(ug, psolar_numpy)
            # print(f"Iteration i, gSCR: {gscr_new}, ug: {ug}")
            if (gscr_ori - self.gscr_threshold) * (gscr_new - self.gscr_threshold) < 0:
                # change sign
                break

        return ug, ug_prev
    
    def perturb_data(self, ug_numpy, psolar_numpy, on_cost, lr, max_iter = 20, verbose = False):
        """
        Perturb both ug and psolar to change the gscr from stable to unstable or vice versa
        """
        ug_new = []
        psolar_new = []

        # Perturb psolar
        psolar_change, psolar_keep, iter = self.perturb_solar(ug_numpy, psolar_numpy, lr, max_iter)
        ug_new.append(ug_numpy)
        psolar_new.append(psolar_change)
        ug_new.append(ug_numpy)
        psolar_new.append(psolar_keep)

        if verbose:
            gscr_ori = self.compute_gSCR(ug_numpy, psolar_numpy)
            gscr_psolar_change = self.compute_gSCR(ug_numpy, psolar_change)
            gscr_psolar_keep = self.compute_gSCR(ug_numpy, psolar_keep)

            print(f"Iter no. for perturbing solar: {iter}")
            print(f"Perturb solar only: gSCR from {gscr_ori} to {gscr_psolar_change} (changed), {gscr_psolar_keep} (kept)")
        
        # Perturb ug
        ug_change, ug_keep = self.perturb_ug(ug_numpy, psolar_numpy, on_cost)
        ug_new.append(ug_change)
        psolar_new.append(psolar_numpy)
        ug_new.append(ug_keep)
        psolar_new.append(psolar_numpy)
        if verbose:
            gscr_ug_change = self.compute_gSCR(ug_change, psolar_numpy)
            gscr_ug_keep = self.compute_gSCR(ug_keep, psolar_numpy)
        
            print(f"Perturb ug only: gSCR from {gscr_ori} to {gscr_ug_change} (changed), {gscr_ug_keep} (kept)")
        
        # Perturb psolar with ug_change and ug_keep
        psolar_change2, psolar_keep2, iter2 = self.perturb_solar(ug_change, psolar_numpy,
                                                            lr=lr, max_iter=max_iter)
        psolar_change3, psolar_keep3, iter3 = self.perturb_solar(ug_keep, psolar_numpy,
                                                            lr=lr, max_iter=max_iter)
        ug_new.append(ug_change)
        psolar_new.append(psolar_change2)
        ug_new.append(ug_change)
        psolar_new.append(psolar_keep2)
        ug_new.append(ug_keep)
        psolar_new.append(psolar_change3)
        ug_new.append(ug_keep)
        psolar_new.append(psolar_keep3)
        if verbose:
            gscr_psolar_change2 = self.compute_gSCR(ug_change, psolar_change2)
            gscr_psolar_keep2 = self.compute_gSCR(ug_change, psolar_keep2)
            gscr_psolar_change3 = self.compute_gSCR(ug_keep, psolar_change3)
            gscr_psolar_keep3 = self.compute_gSCR(ug_keep, psolar_keep3)

            print(f"Iter no. for perturbing solar with ug_change: {iter2}")
            print(f"Iter no. for perturbing solar with ug_keep: {iter3}")
            print(f"Perturb solar with ug_change: gSCR from {gscr_ug_change} to {gscr_psolar_change2} (changed), {gscr_psolar_keep2} (kept)")
            print(f"Perturb solar with ug_keep: gSCR from {gscr_ug_keep} to {gscr_psolar_change3} (changed), {gscr_psolar_keep3} (kept)")
            
        return np.array(ug_new), np.array(psolar_new)


def _build_sorted_bbus_from_raw(uc, bbus_raw):
    """Reorder a raw Bbus into [renewable, generator-bus, other] ordering."""
    new_identity = np.zeros((uc.no_bus, uc.no_bus))
    for i, idx in enumerate(uc.solar_idx):
        new_identity[idx, i] = 1

    start_idx = uc.no_solar
    checked_gen_idx = []
    i = 0
    for idx in uc.gen_idx:
        if idx not in checked_gen_idx:
            new_identity[idx, i + start_idx] = 1
            i += 1
            checked_gen_idx.append(idx)

    start_idx += len(checked_gen_idx)
    for i, idx in enumerate(uc.not_gen_ren_idx):
        new_identity[idx, i + start_idx] = 1

    bbus_sorted = new_identity.T @ bbus_raw @ new_identity
    return bbus_sorted


def compute_gscr_with_min_x_trip(uc, xd, gscr_threshold, solar_min_clip, solar_max, ug, psolar):
    """
    Compute base-case gSCR and min-reactance-line-trip gSCR for the same (ug, psolar).

    Returns:
        dict with keys:
            gscr_base
            gscr_trip
            trip_idx_original
            trip_reactance
            tripped_line
    """
    sss_base = SmallSignalStability(uc, xd, gscr_threshold, solar_min_clip, solar_max)
    gscr_base = sss_base.compute_gSCR(ug, psolar)

    topo_trip = uc.build_min_reactance_trip_topology()
    bbus_trip_sorted = _build_sorted_bbus_from_raw(uc, topo_trip["Bbus"])

    sss_trip = deepcopy(sss_base)
    sss_trip.Bbus_sorted = bbus_trip_sorted
    gscr_trip = sss_trip.compute_gSCR(ug, psolar)

    return {
        "gscr_base": gscr_base,
        "gscr_trip": gscr_trip,
        "trip_idx_original": topo_trip["trip_idx_original"],
        "trip_reactance": topo_trip["trip_reactance"],
        "tripped_line": topo_trip["tripped_line"],
    }


def train_logistic_assessor(input_space, label, type = 'linear_cbce'):
    """
    type:
        linear_cbce: linear classifier with conservative binary cross entropy loss
        linear_bce: linear classifier with binary cross entropy loss
    """
    W_assessor = cp.Variable(input_space.shape[1], name = 'W_assessor')
    b_assessor = cp.Variable((), name = 'b_assessor')
    unstable_idx = np.where(label == 1)[0]
    stable_idx = np.where(label == 0)[0]
    
    if type == 'linear_cbce':
        # only maximize the entropy of the stable samples
        obj = -cp.sum(cp.logistic(input_space[stable_idx] @ W_assessor + b_assessor))
        constraints = [
            input_space[unstable_idx] @ W_assessor + b_assessor >= 1e-3, 
            # unstable sample must be correctly classified
        ]
        prob = cp.Problem(cp.Maximize(obj), constraints) # Conservative constraint
        
    elif type == 'linear_bce':
        # standard BCE with logits loss
        obj = -cp.sum(
            cp.logistic(input_space @ W_assessor + b_assessor)
            ) + cp.sum(
                input_space[unstable_idx] @ W_assessor + b_assessor
                )
        prob = cp.Problem(cp.Maximize(obj))  # Unconstrained
    else:
        raise ValueError(f"Invalid type: {type}")
        
    prob.solve(solver = cp.CLARABEL)
    
    print(f"Status: {prob.status}")
    
    W_assessor = W_assessor.value
    b_assessor = b_assessor.value
    
    # Accuracy of the logistic regression model
    pred_label = np.where(input_space @ W_assessor + b_assessor >= 0, 1, 0)
    TPR = np.sum(pred_label[unstable_idx] == 1) / len(unstable_idx)
    FPR = np.sum(pred_label[stable_idx] == 1) / len(stable_idx)
    print(f"TPR (should be exactly 1 for Conservative BCE): {TPR}, FPR: {FPR}")
    
    return W_assessor, b_assessor 

def return_nn(input_shape, type = 'nn_small'):
    if type == 'nn_small':
        model = nn.Sequential(
            nn.Linear(input_shape, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
    elif type == 'nn_medium':
        model = nn.Sequential(
            nn.Linear(input_shape, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    elif type == 'nn_large':
        model = nn.Sequential(
            nn.Linear(input_shape, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(), 
            nn.Linear(5, 1)    
        )
    elif type == 'nn_very_large': # not that large....
        model = nn.Sequential(
            nn.Linear(input_shape, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(), 
            nn.Linear(10, 1)    
        )
        
    # New larger models
    elif type == "nn_large_1":
        model = nn.Sequential(
            nn.Linear(input_shape, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(), 
            nn.Linear(30, 1)    
        )
    elif type == "nn_large_2":
        model = nn.Sequential(
            nn.Linear(input_shape, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(), 
            nn.Linear(50, 1)    
        )
    elif type == "nn_large_3":
        model = nn.Sequential(
            nn.Linear(input_shape, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(), 
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)    
        )
    else:
        raise ValueError(f"Invalid model type: {type}")
    
    return model

def train_nn(model, data, label, batch_size = 64, 
            lr = 0.001, max_iter = 100, patience = 10, device = 'cuda:0', regression = False, weight_decay = 0):
    # Binary classification, with <0 as stable
    # Overfit on the training data
    if regression:
        # train to forcast the exact gscr value
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    
    # Define dataset
    dataset = torch.utils.data.TensorDataset(data, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    model.to(device)
    data = data.to(device)
    label = label.to(device)
    
    for epoch in range(max_iter):
        for i, (input_sample, label_sample) in enumerate(dataloader):
            optimizer.zero_grad()
            input_sample = input_sample.to(device)
            label_sample = label_sample.to(device)
            output = model(input_sample)
            loss = criterion(output.flatten(), label_sample)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            loss_eval = 0
            acc_eval = 0
            no_sample = 0
            for i, (input_sample, label_sample) in enumerate(dataloader):
                input_sample = input_sample.to(device)
                label_sample = label_sample.to(device)
                output = model(input_sample)
                loss = criterion(output.flatten(), label_sample) * len(input_sample)
                # output >0 as unstable
                if regression:
                    predictions = output.flatten()
                    # consider a prediction is correct if the absolute error is less than 0.1
                    acc = (torch.abs(predictions - label_sample) < 0.1).float().sum()
                else:
                    predictions = (output.flatten() > 0).float()
                    acc = (predictions == label_sample).float().sum()
                    
                loss_eval += loss.item()
                acc_eval += acc.item()
                no_sample += len(input_sample)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {np.round(loss_eval / no_sample, 4)}, Accuracy: {np.round(acc_eval / no_sample, 4)}")
        
            # Early stopping
            if loss_eval / no_sample < best_loss:
                best_loss = loss_eval / no_sample
                best_model = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} with best loss {best_loss}")
                model.load_state_dict(best_model)
                return model.to('cpu')
                
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model.to('cpu')

# Solve optimization problem
def assign_parameter(original_problem, original_input):
    for param in original_problem.parameters():
        try:
            param.value = original_input[param.name()]
        except:
            print(f"Parameter {param.name()} not found in original input or the shape mismatches")


def solve_problem(problem, input: dict, threads = 1, concurrent = 1, seed = 0,
                    opt_params = None, solver = None):

    assign_parameter(problem, input)
    if solver == "GUROBI":
        problem.solve(solver = getattr(cp, solver),
                        **opt_params)
    elif solver == "MOSEK":
        if "warm_start" in opt_params:
            with open_dict(opt_params):
                warm_start = opt_params.pop("warm_start")
        else:
            warm_start = True
        if "verbose" in opt_params:
            with open_dict(opt_params):
                verbose = opt_params.pop("verbose")
        else:
            verbose = False
        problem.solve(solver = getattr(cp, solver), 
                        warm_start = warm_start, verbose = verbose,
                        mosek_params = opt_params)
    else:
        raise ValueError(f"Invalid solver: {solver}")

    solution = {var.name(): var.value for var in problem.variables()}
    optimal_value = problem.value
    
    return solution, optimal_value


def evaluate_uc(prob, small_signal_stability, nn, test_sample_idx, load, solar, T, 
                threads = 1, concurrent = 1, seed = 0,
                solver = None, opt_params = None, regression = False,
                log_verbose = True):
    """
    Pass the UC problem (with sco constraints, etc)
    and evaluate the performance
    this is used for the SCO experiment only
    """
    cost_total = []
    gscr_total = []     # true gscr >gscr_threshold is stable, <=gscr_threshold is unstable
    gscr_cls_total = [] # gscr evaluated by the classifier, >0 is unstable, <=0 is stable
    ug_total = []
    ls_total = []
    solarc_total = []
    solar_total = []
    for idx in (tqdm(test_sample_idx) if len(test_sample_idx) > 1 else test_sample_idx):
        
        load_sample = load[idx:idx+T]
        # ! clip the solar power to be positive
        solar_sample = np.clip(solar[idx:idx+T], a_min = small_signal_stability.solar_min_clip, a_max = None)
        param_data = {'load': load_sample, 'solar': solar_sample}
        sol, obj = solve_problem(prob, param_data, threads = threads, concurrent = concurrent, seed = seed,
                                solver = solver, opt_params = opt_params)
        
        ug = np.rint(sol['ug'])
        solarc = sol['solarc']
        ls = sol['ls']
        ug_total.append(ug)
        solarc_total.append(solarc)
        solar_total.append(solar_sample)
        cost_total.append(obj)
        ls_total.append(ls)
        
        # Classifier output
        assert np.all(solarc <= solar_sample), "The solar power is not positive."
        cls_input = np.hstack([ug, solar_sample - solarc])
        cls_output = nn(torch.from_numpy(cls_input).float())
        gscr_cls_total.append(cls_output.flatten().tolist())
        
        if not np.all(np.sum(ug, axis = 1) >= 1):
            print(f"At least one generator is off at sample {idx}")
            print(f"ug: {np.sum(ug, axis = 1)}")
            print(f"solar_sample: {solar_sample}")
            print(f"solarc: {solarc}")
            exit()
        
        # True gscr
        gscr_per_day = []
        for t in range(T):
            gscr_true = small_signal_stability.compute_gSCR(ug[t], solar_sample[t] - solarc[t])
            gscr_per_day.append(gscr_true)
        gscr_total.append(gscr_per_day)

    gscr_total = np.array(gscr_total)  # (no_days, T)
    gscr_cls_total = np.array(gscr_cls_total)  # (no_days, T)
    cost_total = np.array(cost_total)
    ug_total = np.array(ug_total)  # (no_days, T, no_gen)
    solarc_total = np.array(solarc_total)
    solar_total = np.array(solar_total)
    ls_total = np.array(ls_total)
    
    cost_ave = np.mean(cost_total) * 100 # per day
    # On average, how many hours are unstable per day
    unstable_hours_per_day_ave = np.mean(np.sum(gscr_total <= small_signal_stability.gscr_threshold, axis = 1))
    if regression:
        unstable_hours_per_day_ave_cls = np.mean(np.sum(gscr_cls_total <= small_signal_stability.gscr_threshold, axis = 1))
    else:
        unstable_hours_per_day_ave_cls = np.mean(np.sum(gscr_cls_total > 0, axis = 1))
    # Sum up the number of unstable days: how many days are unstable
    no_unstable_days_ratio = np.sum(np.sum(gscr_total <= small_signal_stability.gscr_threshold, axis = 1) > 0) / len(test_sample_idx)
    if regression:
        no_unstable_days_cls_ratio = np.sum(np.sum(gscr_cls_total <= small_signal_stability.gscr_threshold, axis = 1) > 0) / len(test_sample_idx)
    else:
        no_unstable_days_cls_ratio = np.sum(np.sum(gscr_cls_total > 0, axis = 1) > 0) / len(test_sample_idx)
    
    if log_verbose:
        print(f"Average cost: {cost_ave}")
        print(f"Average unstable hours per day: {unstable_hours_per_day_ave}") 
        print(f"Average unstable hours per day (cls): {unstable_hours_per_day_ave_cls}")
        print(f"Number of unstable days ratio: {no_unstable_days_ratio}")  # the ratio of days with at least one unstable hour
        print(f"Number of unstable days ratio (cls): {no_unstable_days_cls_ratio}")
    
    return cost_total, gscr_total, gscr_cls_total, ug_total, solarc_total, solar_total, ls_total


def evaluate_uc_basic(prob, small_signal_stability, test_sample_idx, load, solar, T,
                    threads = 1, concurrent = 1, seed = 0, solver = None,
                    opt_params = None):
    """
    Use this function before a classifier is specified
    """

    cost_total = []
    gscr_total = []     # true gscr >gscr_threshold is stable, <=gscr_threshold is unstable
    ug_total = []
    ls_total = []
    solarc_total = []
    solar_total = []

    for idx in tqdm(test_sample_idx):
        
        load_sample = load[idx:idx+T]
        # ! clip the solar power to be positive
        solar_sample = np.clip(solar[idx:idx+T], a_min = small_signal_stability.solar_min_clip, a_max = None)
        param_data = {'load': load_sample, 'solar': solar_sample}
        sol, obj = solve_problem(prob, param_data, threads = threads, concurrent = concurrent, seed = seed,
                                solver = solver,
                                opt_params = opt_params)
        
        ug = np.rint(sol['ug'])
        solarc = sol['solarc']
        ls = sol['ls']
        ug_total.append(ug)
        solarc_total.append(solarc)
        solar_total.append(solar_sample)
        cost_total.append(obj)
        ls_total.append(ls)
        
        if not np.all(np.sum(ug, axis = 1) >= 1):
            print(f"At least one generator is off at sample {idx}")
            print(f"ug: {np.sum(ug, axis = 1)}")
            print(f"solar_sample: {solar_sample}")
            print(f"solarc: {solarc}")
            exit()
        
        # True gscr
        gscr_per_day = []
        for t in range(T):
            gscr_true = small_signal_stability.compute_gSCR(ug[t], solar_sample[t] - solarc[t])
            gscr_per_day.append(gscr_true)
        gscr_total.append(gscr_per_day)

    gscr_total = np.array(gscr_total)  # (no_days, T)
    cost_total = np.array(cost_total)
    ug_total = np.array(ug_total)
    solarc_total = np.array(solarc_total)
    solar_total = np.array(solar_total)
    ls_total = np.array(ls_total)
    
    cost_ave = np.mean(cost_total) * 100 # per day
    # On average, how many hours are unstable per day
    unstable_hours_per_day_ave = np.mean(np.sum(gscr_total <= small_signal_stability.gscr_threshold, axis = 1))
    # Sum up the number of unstable days: how many days are unstable
    no_unstable_days_ratio = np.sum(np.sum(gscr_total <= small_signal_stability.gscr_threshold, axis = 1) > 0) / len(test_sample_idx)
    
    print(f"Average cost: {cost_ave}")
    print(f"Average unstable hours per day: {unstable_hours_per_day_ave}") 
    print(f"Number of unstable days ratio: {no_unstable_days_ratio}")  # the ratio of days with at least one unstable hour
    
    return cost_total, gscr_total, ug_total, solarc_total, solar_total, ls_total
