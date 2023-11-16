import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def hook_fn(module, input, output):
    module.output = output

def register_single_hook(model, layer_idx, hook_fn):
    """
    Register a hook to a single layer given by index.

    Parameters:
    model (torch.nn.Module): The model to which we are registering the hook.
    layer_idx (int): The index of the layer to which we want to attach the hook.
    hook_fn (function): The hook function that will be called during the forward pass.
    """
    # Initialize a counter to keep track of the layer index
    current_idx = 0

    # Define a recursive function to iterate over the model's children
    def _recursive_register_hook(module):
        nonlocal current_idx
        for child in module.children():
            if len(list(child.children())) == 0:  # If it is a leaf module
                if current_idx == layer_idx:
                    child.register_forward_hook(hook_fn)
                    return True
                current_idx += 1
            else:  # If module has children, recursively proceed to add hooks
                found = _recursive_register_hook(child)
                if found:  # If we've found and registered the hook, we stop looking
                    return True
        return False

    # Start the recursive hook registration process
    _recursive_register_hook(model)

def load_model(model_name, path=None):
    if model_name == 'mobilenet':
        model = mobilenet_v2(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier._modules["1"].in_features
        model.classifier._modules["1"] = torch.nn.Linear(num_ftrs, 10)
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        if path is not None:
            model.load_state_dict(torch.load(path))

    return model.eval()

def return_feature_map(model, layer_idx):
    """
    Register a hook to a single layer given by index.

    Parameters:
    model (torch.nn.Module): The model to which we are registering the hook.
    layer_idx (int): The index of the layer to which we want to attach the hook.
    hook_fn (function): The hook function that will be called during the forward pass.
    """
    # Initialize a counter to keep track of the layer index
    current_idx = 0
    feature_map = None

    # Define a recursive function to iterate over the model's children
    def _recursive_collect_output(module):
        nonlocal current_idx, feature_map
        for child in module.children():
            if len(list(child.children())) == 0:  # If it is a leaf module
                if current_idx == layer_idx:
                    feature_map = child.output.cpu()
                    return True
                current_idx += 1
            else:  # If module has children, recursively proceed to add hooks
                found = _recursive_collect_output(child)
                if found:  # If we've found and registered the hook, we stop looking
                    return True
        return False


    

    # Start the recursive hook registration process
    _recursive_collect_output(model)

    return feature_map

def print_nested_info(obj, depth=0):
    if isinstance(obj, (list, tuple)):
        print(f"Depth {depth}: {type(obj).__name__} with length = {len(obj)}")
        if len(obj) > 0:
            print_nested_info(obj[0], depth + 1)
    else:
        print(f"Depth {depth}: Element type = {type(obj)}")