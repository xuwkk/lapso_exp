import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import numpy as np

def extract_layers_list(model):
    """Extracts all layers as a flat list.
    Args:
        model: Sequential neural network model or model defined by nn.ModuleList
    Returns:
        List of all layers in the model
    """
    layers = []
    for _, module in model.named_modules():
        if not list(module.children()):  # Avoid adding parent modules
            layers.append(module)
    return layers

def _layer_info(model, sample_input):
    """
    Get the layer information for the model
    Args:
        model: Sequential neural network model or model defined by nn.ModuleList
        sample_input: Input tensor
    Returns:
        layer_info: List information including {layer_type, output_size, binary_vars} per layer
    """
    outputs, output_sizes = get_layer_output(sample_input, model)
    output_sizes = [size[1:] for size in output_sizes] # remove the batch dimension
    layer_info = []
    total_binary = 0
    
    for i, layer in enumerate(model):
        layer_dict = {
            'layer_type': type(layer).__name__,
            'output_size': None,
            'binary_vars': 0
        }
        
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer_dict['output_size'] = output_sizes[i]
                
        elif isinstance(layer, nn.MaxPool2d):
            layer_dict['output_size'] = output_sizes[i]
            C, H, W = output_sizes[i]
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            layer_dict['binary_vars'] = C * H * W * k * k
            total_binary += layer_dict['binary_vars']
            
        elif isinstance(layer, nn.ReLU):
            layer_dict['output_size'] = output_sizes[i-1]  # ReLU keeps same size as previous layer
            # Add binary variables if previous layer was Conv2d or Linear
            if isinstance(model[i-1], (nn.Linear, nn.Conv2d)):
                layer_dict['binary_vars'] = np.prod(output_sizes[i-1])
                total_binary += layer_dict['binary_vars']
            
        elif isinstance(layer, nn.Flatten):
            layer_dict['output_size'] = output_sizes[i]
            
        layer_info.append(layer_dict)
        
    return layer_info, total_binary

def bound_propagation(model, initial_bound):
    """Interval Bound Propagation (IBP) for neural network layers.
    
    Computes output bounds for flatten, linear, conv2d, max-pooling layers with ReLU activation.
    
    Args:
        model: Sequential neural network model
        initial_bound: Tuple of (lower_bound, upper_bound) tensors,
            assuming a Linf norm bound,
            with shape (batch, channels, height, width) or (batch, features)
        
    Returns:
        List of (lower_bound, upper_bound) tuples for each layer's output
    """
    l, u = initial_bound
    
    bounds = []
    
    for layer in model:
        # Flatten layer
        if isinstance(layer, nn.Flatten):
            l_ = layer(l)
            u_ = layer(u)
            
        # Linear layer    
        elif isinstance(layer, nn.Linear):
            w_pos = layer.weight.clamp(min=0)
            w_neg = layer.weight.clamp(max=0)
            l_ = l @ w_pos.t() + u @ w_neg.t() + layer.bias[None,:]
            u_ = u @ w_pos.t() + l @ w_neg.t() + layer.bias[None,:]
                        
        # Conv2d layer
        elif isinstance(layer, nn.Conv2d):
            w_pos = layer.weight.clamp(min=0) 
            w_neg = layer.weight.clamp(max=0)
            
            conv_kwargs = {
                'stride': layer.stride,
                'padding': layer.padding,
                'dilation': layer.dilation,
                'groups': layer.groups
            }
            
            l_ = (F.conv2d(l, w_pos, bias=None, **conv_kwargs) +
                 F.conv2d(u, w_neg, bias=None, **conv_kwargs) +
                layer.bias[None,:,None,None])
                
            u_ = (F.conv2d(u, w_pos, bias=None, **conv_kwargs) +
                 F.conv2d(l, w_neg, bias=None, **conv_kwargs) + 
                layer.bias[None,:,None,None])
            
        # MaxPool2d layer
        elif isinstance(layer, nn.MaxPool2d):
            l_ = F.max_pool2d(l, kernel_size=layer.kernel_size, 
                            stride=layer.stride, padding=layer.padding)
            u_ = F.max_pool2d(u, kernel_size=layer.kernel_size,
                            stride=layer.stride, padding=layer.padding)
            
        # ReLU layer
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
            
        bounds.append((l_, u_))
        l, u = l_, u_
        
    return bounds

def pad_image_with_matrix(x, row_padding=0, col_padding=0, dtype="float32"):
    """Pad an image using matrix multiplication A * x * B to simulate separate row and column padding.
    
    Args:
        x: Input image with shape (channels, height, width)
        row_padding: Padding to apply to the rows (top and bottom)
        col_padding: Padding to apply to the columns (left and right) 
        dtype: Data type for the padding matrices
        
    Returns:
        tuple: Padding matrices (A, B) where:
            A has shape (channels, padded_height, height)
            B has shape (channels, padded_width, width)
    """
    assert x.ndim == 3, "Input must be 3D (C_in, H, W)"
    channels, height, width = x.shape
    
    # Calculate padded dimensions
    padded_height = height + 2 * row_padding
    padded_width = width + 2 * col_padding

    # Create padding matrices based on input type
    if isinstance(x, torch.Tensor):
        dtype = getattr(torch, dtype)
        A = torch.zeros(padded_height, height, device=x.device, dtype=dtype)
        B = torch.zeros(padded_width, width, device=x.device, dtype=dtype)
    else:
        # for both numpy and cvxpy
        dtype = getattr(np, dtype)
        A = np.zeros((padded_height, height), dtype=dtype)
        B = np.zeros((padded_width, width), dtype=dtype)

    # Fill in the identity mapping positions
    for i in range(height):
        A[i + row_padding, i] = 1
    for i in range(width):
        B[i + col_padding, i] = 1
        
    return A, B

def manual_im2col(X, kernel_size=2, stride=1, padding=0, dtype="float32"):
    """
    Manually implement unfold operation similar to torch.nn.functional.unfold
    using matrix multiplication with minimum number of reshapes.
    TODO: Support dilation
    
    Args:
        X (tensor/array/cvxpy): Input of shape (channels, height, width)
        kernel_size (int or tuple): Size of the sliding window
        stride (int or tuple): Stride of the sliding window
        padding (int or tuple): Padding size
    Returns:
        tensor/array/cvxpy: Unfolded tensor of shape (C * k_h * k_w, H_out * W_out)
        where H_out = (H + 2*padding[0] - k_h)//stride[0] + 1
        and W_out = (W + 2*padding[1] - k_w)//stride[1] + 1
    """
    assert X.ndim == 3, "Input must be a 3D tensor"
    
    # Convert scalar inputs to tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Handle padding
    if padding[0] > 0 or padding[1] > 0:
        A, B = pad_image_with_matrix(X, padding[0], padding[1])
        if isinstance(X, (np.ndarray, torch.Tensor)):
            X_padded = A @ X @ B.T
        else:
            # for cvxpy
            # X_padded = cp.matmul(cp.matmul(A, X), B.T)
            # NOTE: the cvxpy variable does not support propagation?
            padded_channels = [cp.matmul(cp.matmul(A, X[c]), B.T)[None] for c in range(X.shape[0])]
            X_padded = cp.vstack(padded_channels)
    else:
        X_padded = X
        
    # Calculate output dimensions
    channels, height, width = X_padded.shape
    out_height = (height - kernel_size[0]) // stride[0] + 1
    out_width = (width - kernel_size[1]) // stride[1] + 1
    
    # Flatten input
    if isinstance(X_padded, (torch.Tensor, np.ndarray)):
        X_flatten = X_padded.reshape(-1)
    else:
        # for cvxpy
        X_flatten = X_padded.reshape(-1, order='C')
    
    # Create transformation matrix
    transform_size = (channels * kernel_size[0] * kernel_size[1] * out_height * out_width,
                     channels * height * width)
    
    if isinstance(X_padded, torch.Tensor):
        dtype = getattr(torch, dtype)
        transform_matrix = torch.zeros(transform_size, dtype=dtype)
    else:
        transform_matrix = np.zeros(transform_size, dtype=dtype)
    
    # Fill transformation matrix
    for c in range(channels):
        for i in range(out_height):
            for j in range(out_width):
                for ki in range(kernel_size[0]):
                    for kj in range(kernel_size[1]):
                        input_idx = c * height * width + (i * stride[0] + ki) * width + (j * stride[1] + kj)
                        output_idx = ((i * out_width + j) * channels * kernel_size[0] * kernel_size[1] + 
                                    c * kernel_size[0] * kernel_size[1] + 
                                    ki * kernel_size[1] + kj)
                        transform_matrix[output_idx, input_idx] = 1
    
    # Apply transformation and reshape result
    result = transform_matrix @ X_flatten
    if isinstance(X_flatten, (torch.Tensor, np.ndarray)):
        return result.reshape(out_height * out_width, 
                            channels * kernel_size[0] * kernel_size[1]).T
    else:
        return result.reshape((out_height * out_width,
                            channels * kernel_size[0] * kernel_size[1]), # for cvxpy
                            order='C').T

def get_layer_output(X, model):
    """Get the output size and values for each layer in the original model.
    
    Args:
        X: Input tensor
        model: Neural network model
        
    Returns:
        output_sizes: List of output shapes for each layer
        outputs: List of output tensors for each layer
    """
    output = X.clone()
    output_sizes = []
    outputs = []
    
    for layer in model:
        output = layer(output)
        output_sizes.append(output.shape)
        outputs.append(output.clone())
            
    return outputs, output_sizes

def get_layer_output_unfold(X, model):
    """Forward pass through model using matrix form unfolding and convolution.
    
    Args:
        X: Input tensor
        model: Neural network model
        
    Returns:
        List of output tensors for each layer
    """
    outputs, output_sizes = get_layer_output(X[None], model)
    
    # unfold the output
    output_unfold_all = []
    output_unfold = X.clone()
    
    for idx, layer in enumerate(model):
        # Conv2d layer
        if isinstance(layer, nn.Conv2d):
            output_unfold = manual_im2col(output_unfold, kernel_size=layer.kernel_size, 
                                stride=layer.stride, padding=layer.padding)
            K = layer.weight.view(layer.out_channels, -1)
            
            # forward the unfolded input and weight
            H_out = output_sizes[idx][2]
            W_out = output_sizes[idx][3]
            output_unfold = K @ output_unfold + layer.bias.view(-1, 1)
            output_unfold = output_unfold.view(layer.out_channels, H_out, W_out) # reshape back to original shape
        
        # the following layers are the same as the original forward pass
        # Linear layer
        elif isinstance(layer, nn.Linear):
            output_unfold = layer.weight @ output_unfold + layer.bias
            
        # ReLU layer
        elif isinstance(layer, nn.ReLU):
            output_unfold = output_unfold.clamp(min=0)
            
        # Flatten layer
        elif isinstance(layer, nn.Flatten):
            output_unfold = output_unfold.flatten(start_dim=0)
        
        # MaxPool2d layer
        elif isinstance(layer, nn.MaxPool2d):
            output_unfold = F.max_pool2d(output_unfold, kernel_size=layer.kernel_size, 
                            stride=layer.stride, padding=layer.padding)
            
        output_unfold_all.append(output_unfold)
            
    return output_unfold_all
    
def form_milp_(model, initial_bound, bounds, verbose=False):
    """
    Formulate the exact adversarial attack as MILP.
    Supports conv2d, linear, flatten, and ReLU layers.
    TODO: Support dilation in conv2d layers

    Args:
        model: Neural network model
        c: Target vector for optimization objective
        initial_bound: Tuple of (lower, upper) bounds on input
        bounds: List of (lower, upper) bounds for each layer output
        
    Returns:
        constraints: List of CVXPY constraints
        variables: Tuple of (z, v) variables used in formulation
    """
    no_layers = len(bounds)
    
    # Get layer sizes
    outputs, output_sizes = get_layer_output(initial_bound[0], model)
    output_sizes = [size[1:] for size in output_sizes] # remove the batch dimension
    layer_input_size = [initial_bound[0].shape[1:]] + output_sizes[:-1]
    
    # Create variables for each layer outputs
    z = []
    for i in range(no_layers):
        if isinstance(model[i], (nn.Linear, nn.Conv2d, nn.Flatten, nn.MaxPool2d)):
            z.append(cp.Variable(layer_input_size[i]))
    z.append(cp.Variable(output_sizes[-1]))

    # Create binary variables for ReLU
    v = []
    has_relu = []  # Track which layers are followed by ReLU
    i = 0
    while i < no_layers - 1:
        if isinstance(model[i], (nn.Linear, nn.Conv2d)):
            # Check if next layer is ReLU
            if isinstance(model[i + 1], nn.ReLU):
                v.append(cp.Variable(output_sizes[i], boolean=True))
                has_relu.append(True)
                i += 2  # Skip the ReLU layer
            else:
                v.append(None)  # No ReLU, no binary variables needed
                has_relu.append(False)
                i += 1
        elif isinstance(model[i], nn.MaxPool2d):
            # Get dimensions for maxpool binary variables
            C, H, W = output_sizes[i]
            k = model[i].kernel_size if isinstance(model[i].kernel_size, int) else model[i].kernel_size[0]
            # Shape: (C, H_out, W_out, k*k) for each element in each pooling window
            v.append(cp.Variable((C, H, W, k*k), boolean=True))
            has_relu.append(False)  # no relu following after maxpooling
            i += 1
        elif isinstance(model[i], nn.Flatten):
            v.append(None)  # flatten layer does not have any ReLU
            has_relu.append(False)
            i += 1
        else:
            raise NotImplementedError(f"Layer type {type(model[i])} not supported")

    # Get linear layers and parameters
    linear_layers = [(layer, bound) for layer, bound in zip(model, bounds) 
                    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Flatten, nn.MaxPool2d))]

    # Extract weights and biases
    W, b = [], []
    for layer, _ in linear_layers:
        if isinstance(layer, (nn.Flatten, nn.MaxPool2d)):
            W.append(None)
            b.append(None) # flatten layer does not have any weights and biases
        else:
            W.append(layer.weight.detach().cpu().numpy())
            b.append(layer.bias.detach().cpu().numpy())
    
    # Extract bounds
    l = [bound[0][0].detach().cpu().numpy() for _, bound in linear_layers] # remove the batch dimension
    u = [bound[1][0].detach().cpu().numpy() for _, bound in linear_layers]
    l0 = initial_bound[0][0].detach().cpu().numpy() # remove the batch dimension
    u0 = initial_bound[1][0].detach().cpu().numpy()
    
    # Build constraints
    constraints = []
    # j = 0  # Index for linear_layers
    for i in range(len(linear_layers)-1):
        # Linear layer constraints
        if isinstance(linear_layers[i][0], nn.Linear):
            b_ = b[i][None] if z[i].ndim == 2 else b[i]
                
            if has_relu[i]:
                if verbose:
                    print('linear layer with ReLU \n', z[i].shape, '->', z[i+1].shape)
                    print('W and b', W[i].shape, b_.shape)
                    print('v, u, l', v[i].shape, u[i].shape, l[i].shape)
                constraints += [
                    z[i+1] >= z[i] @ W[i].T + b_,
                    z[i+1] >= 0,
                    cp.multiply(v[i], u[i]) >= z[i+1],
                    z[i] @ W[i].T + b_ >= z[i+1] + cp.multiply((1-v[i]), l[i])
                ]
            else:
                # Case without ReLU
                # constraints += [z[i+1] == W[i] @ z[i] + b[i]]
                constraints += [z[i+1] == z[i] @ W[i].T + b_]
                if verbose:
                    print('linear layer without ReLU \n', z[i].shape, '->', z[i+1].shape)
                    print('W and b', W[i].shape, b_.shape)
        
        # MaxPool2d layer constraints
        elif isinstance(linear_layers[i][0], nn.MaxPool2d):
            
            if verbose:
                print('maxpool layer \n', z[i].shape, '->', z[i+1].shape)
                print('z_unfold', z_unfold.shape)
            
            kernel_size = linear_layers[i][0].kernel_size
            stride = linear_layers[i][0].stride
            padding = linear_layers[i][0].padding
            
            # Unfold input for pooling windows
            # NOTE: this has been padded with zeros
            z_unfold = manual_im2col(z[i], kernel_size=kernel_size,
                                   stride=stride, padding=padding) # [C * k*k, H * W]
            # Get output dimensions
            C, height, width = layer_input_size[i]
            # NOTE: dont add padding again
            height_out = (height - kernel_size) // stride + 1
            width_out = (width - kernel_size) // stride + 1
            # Reshape output and binary variables
            z_out = z[i+1].reshape((C, height_out * width_out), order='C')
            v_pool = v[i].reshape((C, height_out * width_out, kernel_size * kernel_size), order='C')
            
            # Use upper bound as Big-M value
            M = u[i].max()
            
            # For each channel and each pooling window
            for c in range(C):
                for j in range(height_out * width_out):
                    # Get the corresponding slice of unfolded input
                    # This is a window slice of the unfolded input based on the pooling window
                    window_slice = z_unfold[c*kernel_size*kernel_size:(c+1)*kernel_size*kernel_size, j]
                    
                    # 1. Sum of binary variables = 1
                    constraints += [cp.sum(v_pool[c,j,:]) == 1]
                    
                    # 2. Output is greater than or equal to all inputs
                    constraints += [z_out[c,j] >= window_slice]
                    
                    # 3. Big-M constraints: output is less than or equal to selected input
                    for k in range(kernel_size * kernel_size):
                        constraints += [z_out[c,j] <= window_slice[k] + M*(1 - v_pool[c,j,k])]
        
        # Conv2d layer constraints
        elif isinstance(linear_layers[i][0], nn.Conv2d):
            # Unfold weight and bias
            w_unfold = W[i].reshape(linear_layers[i][0].out_channels, -1) # [out_channels, in_channels * kernel_h * kernel_w]
            b_unfold = b[i].reshape(-1, 1)
            
            z_unfold = manual_im2col(z[i], kernel_size=linear_layers[i][0].kernel_size,
                                    stride=linear_layers[i][0].stride, padding=linear_layers[i][0].padding)
            
            # Reshape variables for constraints - specify order='C' for row-major
            z_unfold_next = z[i+1].reshape((linear_layers[i][0].out_channels, -1), order='C')  

            if has_relu[i]:
                v_temp = v[i].reshape((linear_layers[i][0].out_channels, -1), order='C')
                u_temp = u[i].reshape((linear_layers[i][0].out_channels, -1), order='C')
                l_temp = l[i].reshape((linear_layers[i][0].out_channels, -1), order='C')
                # Case with ReLU
                constraints += [
                    z_unfold_next >= (w_unfold @ z_unfold + b_unfold),
                    z_unfold_next >= 0,
                    cp.multiply(v_temp, u_temp) >= z_unfold_next,
                    w_unfold @ z_unfold + b_unfold >= z_unfold_next + cp.multiply((1-v_temp), l_temp)
                ]
                if verbose:
                    print('conv2d layer with ReLU \n', z[i].shape, '->', z[i+1].shape)
                    print('w and b', w_unfold.shape, b_unfold.shape)
                    print('v, u, l', v_temp.shape, u_temp.shape, l_temp.shape)
                    print('z_unfold_next', z_unfold_next.shape, 'z_unfold', z_unfold.shape)
            else:
                # Case without ReLU
                constraints += [z_unfold_next == (w_unfold @ z_unfold + b_unfold)]
                if verbose:
                    print('conv2d layer without ReLU \n', z[i].shape, '->', z[i+1].shape)
                    print('w and b', w_unfold.shape, b_unfold.shape)
            
        # Flatten layer constraints
        elif isinstance(linear_layers[i][0], nn.Flatten):
            if verbose:
                print('flatten layer \n', z[i].shape, '->', z[i+1].shape)
            if z[i].ndim == 3:
                constraints += [z[i+1] == z[i].reshape((-1), order='C')]
            else:
                constraints += [z[i+1] == z[i].reshape((z[i+1].shape[0], -1), order='C')]
    
    # Final linear layer constraint
    if not isinstance(linear_layers[-1][0], nn.Linear):
        raise ValueError("The last layer must be linear")
    # constraints += [z[-1] == W[-1] @ z[-2] + b[-1]]
    if verbose:
        print('final linear layer \n', z[-2].shape, '->', z[-1].shape)
        print('W and b', W[-1].shape, b[-1].shape)
        
    b_ = b[-1][None] if z[-2].ndim == 2 else b[-1]
    constraints += [z[-1] == z[-2] @ W[-1].T + b_]

    constraints += [z[0] >= l0, z[0] <= u0]
    
    # Dont define the problem here, let the user defines the objective function outside
    return constraints, (z, v)
    # return cp.Problem(cp.Minimize(c @ z[-1]), constraints), (z, v)
    
def form_milp_linear(model, c, initial_bound, bounds):
    """Formulate exact accuracy-based attack as MILP for linear layers only.
    
    Args:
        model: Neural network model (linear layers only)
        c: Target vector for optimization objective
        initial_bound: Tuple of (lower, upper) bounds on input
        bounds: List of (lower, upper) bounds for each layer
        
    Returns:
        problem: CVXPY optimization problem
        variables: Tuple of (z, v) variables used in formulation
    """
    linear_layers = [(layer, bound) for layer, bound in zip(model,bounds) if isinstance(layer, nn.Linear)]
    d = len(linear_layers)-1 # number of activations
    
    # Create variables
    z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] + 
            [cp.Variable(linear_layers[-1][0].out_features)]) # input size of each layer + the output size of the last layer
    v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]] # binary variable: output size of each layer except the last
    
    # Extract parameters
    W = [layer.weight.detach().cpu().numpy() for layer,_ in linear_layers]
    b = [layer.bias.detach().cpu().numpy() for layer,_ in linear_layers]
    l = [l[0].detach().cpu().numpy() for _, (l,_) in linear_layers]
    u = [u[0].detach().cpu().numpy() for _, (_,u) in linear_layers]
    l0 = initial_bound[0][0].view(-1).detach().cpu().numpy() # flatten
    u0 = initial_bound[1][0].view(-1).detach().cpu().numpy()
    
    # add ReLU constraints
    constraints = []
    for i in range(len(linear_layers)-1):
        constraints += [z[i+1] >= W[i] @ z[i] + b[i], 
                        z[i+1] >= 0,
                        cp.multiply(v[i], u[i]) >= z[i+1],
                        W[i] @ z[i] + b[i] >= z[i+1] + cp.multiply((1-v[i]), l[i])]
    
    # Final linear constraint
    constraints += [z[d+1] == W[d] @ z[d] + b[d]]
    
    # Input bound constraints
    constraints += [z[0] >= l0, z[0] <= u0]
    
    return cp.Problem(cp.Minimize(c @ z[d+1]), constraints), (z, v)
