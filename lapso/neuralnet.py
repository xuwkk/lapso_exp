"""
high level functions for neural network tools
"""
from lapso.neuralnet_funcs import extract_layers_list, _layer_info, form_milp_, bound_propagation
import numpy as np
import torch

def layer_info(model, sample_input):
    """
    Get the layer information for the model
    Args:
        model: Sequential neural network model or model defined by nn.ModuleList
        sample_input: Input tensor
    Returns:
        layer_info: List information including {layer_type, output_size, binary_vars} per layer
    """
    layer_list = extract_layers_list(model)
    return _layer_info(layer_list, sample_input)

def form_milp(model, initial_bound, verbose=False):
    """
    Form the MILP for the model
    Args:
        model: Sequential neural network model or model defined by nn.ModuleList
        initial_bound: Tuple of (lower_bound, upper_bound) tensors,
            assuming a Linf norm bound,
        verbose: Whether to print the verbose output
    Returns:
        constraints: List of constraints
        (z, v): Tuple of (continuous variables as input and each layer output, 
                    binary variables as indicator of ReLU activation)
    """
    if isinstance(initial_bound[0], np.ndarray):
        initial_bound = (torch.from_numpy(initial_bound[0]).float(), torch.from_numpy(initial_bound[1]).float())
    layer_list = extract_layers_list(model)
    bounds = bound_propagation(layer_list, initial_bound)
    constraints, (z, v) = form_milp_(layer_list, initial_bound, bounds, verbose)
    return constraints, (z, v)
    