"""
Train NN-based ABF
"""

import sys
sys.path.append('./')
import hydra
from omegaconf import DictConfig
from pso.data import get_dataset_np
from pso import prepare_grid_from_pypower, UC_CONTINUOUS, RD
from lapso.optimization import form_kkt, return_standard_form
from obf_func import data_preprocess
import cvxpy as cp
import numpy as np
import os
import torch
from copy import deepcopy
import torch.nn as nn

# Move MLP class definition outside of any function
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, output_dim)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        output = self.fc3(x)
        return output, x  # Also return the hidden layer output
    
def train_acc_forecaster(model, feature, solar, device, batch_size = 64, 
                         lr = 0.001, max_iter = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
    
    # Define dataset
    dataset = torch.utils.data.TensorDataset(feature, solar)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    solar = solar.to(device)
    
    model.to(device)
    for epoch in range(max_iter):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, hidden_layer = model(inputs)
            loss = torch.mean(torch.abs(outputs - labels) / labels) # MAPE
            loss.backward()
            optimizer.step()
        
        # Test the model on the whole dataset
        with torch.no_grad():
            outputs, hidden_layer = model(feature.to(device))
            loss = torch.mean(torch.abs(outputs - solar) / solar) # MAPE
            if loss < best_loss:
                best_loss = loss
                best_model = deepcopy(model.state_dict())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {np.round(loss.cpu().item(), 4)}, Best loss: {np.round(best_loss.cpu().item(), 4)}")
    
    # model.load_state_dict(best_model)
    # return model
    
    return best_model

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    saving_dir = cfg.exp.saving_dir
    train_config = cfg.exp.train_config
    os.makedirs(saving_dir, exist_ok=True)
    
    # Grid
    grid_xlsx = prepare_grid_from_pypower(cfg.grid)
    
    # Load data
    feature, load, solar, _ = get_dataset_np(cfg.grid)
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    feature, load, solar = data_preprocess(feature, load, solar) # Full dataset
    print(f"feature: {feature.shape}, load: {load.shape}, solar: {solar.shape}")
    
    model = MLP(feature.shape[1], solar.shape[1])
    saving_model_path = saving_dir + "acc_forecaster.pth"
    
    if os.path.exists(saving_model_path):
        print("Load the pre-trained model. If you want to train the model from scratch, please delete the file: ", saving_model_path)
    else:
        print('===Train the ABF forecaster===')
        model_state_dict = train_acc_forecaster(model, torch.from_numpy(feature).float(), torch.from_numpy(solar).float(), 
                                     device = DEVICE,
                                     **train_config)
        torch.save(model_state_dict, saving_model_path)
    model.load_state_dict(torch.load(saving_model_path))
    model.to('cpu').eval()
        
    print('===Test the ABF forecaster===')
    feature = torch.from_numpy(feature).float()
    solar = torch.from_numpy(solar).float()
    output, _ = model(feature)
    mape = torch.mean(torch.abs(output - solar) / solar) # MAE
    nrmse = torch.mean(torch.sqrt(torch.mean((output - solar) ** 2, dim = 0)) / torch.std(solar, dim = 0))
    print(f"MAPE: {np.round(mape.item(), 4)}, NRMSE: {np.round(nrmse.item(), 4)}")
    
if __name__ == "__main__":
    main()
