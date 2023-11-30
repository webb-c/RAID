import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms

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


def image_to_frequency(image):
    ft = np.fft.fft2(image)
    ft_shift = np.fft.fftshift(ft)
    magnitude_spectrum = 20*np.log(np.abs(ft_shift))
    return ft_shift, magnitude_spectrum


def frequency_to_image(frequency):
    ft = np.fft.ifftshift(frequency)
    image_back = np.fft.ifft2(ft)
    image = np.abs(image_back)
    return image


def color_image_to_frequency(image):
    """object: 입력 이미지를 frequency domain으로 변환합니다.
        - input: (3, 32, 32) <class 'numpy.ndarry'>"""
    red_tf, red_m = image_to_frequency(image[0,:,:])
    green_tf, green_m = image_to_frequency(image[1,:,:])
    blue_tf, blue_m = image_to_frequency(image[2,:,:])
    frequency = [red_tf, green_tf, blue_tf]
    magnitude = [red_m, green_m, blue_m]
    return frequency, magnitude


def color_frequency_to_image(frequency):
    """object: frequency domain이미지를 원본 이미지 domain으로 변환합니다."""
    red = frequency_to_image(frequency[0])
    green = frequency_to_image(frequency[1])
    blue = frequency_to_image(frequency[2])
    image = np.stack((red, green, blue), axis=0)
    return image


def create_circular_mask(h, w, radius=None, center=None,):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_filtered_image(image, r):
    origin_image = torch.tensor(image)
    c, w, h = origin_image.shape
    mask = create_circular_mask(h, w, radius=r)
    frequency, magnitude = color_image_to_frequency(origin_image)
    masked_frequency = []
    for ft in frequency:
        f_masked = ft * mask
        masked_frequency.append(f_masked)
    next_image = color_frequency_to_image(masked_frequency)
    return next_image