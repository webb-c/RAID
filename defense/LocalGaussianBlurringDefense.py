from defense.DefenseBase import DefenseBase
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal

class LocalGaussianBlurringDefense(DefenseBase):

    def __init__(self, config) -> None:
        self.config = config
    
    def apply(self, image, action):
        """
        LocalGaussianBlurringDefense는 주어진 target pixel(index) 주위에 std만큼의 gaussian blur를 적용합니다.

        input:
            - channel (int): 변형할 이미지의 channel (0: R, 1: G, 2: B)
            - index (int): 변형할 image의 index vector
            - std (float): defense perturbation의 standard deviation
        """
        channel, index, std = action

        y = index // self.config["image_height"]
        x = index % self.config["image_width"]
        kernel_radius = int(self.config["image_height"] * std) + 1
        kernel_size = kernel_radius * 2 + 1
        considered_radius = kernel_radius * 2

        # Patch coordinate with form (y1, x1, y2, x2)
        boundary = (
            max(y - considered_radius, 0),
            max(x - considered_radius, 0),
            min(y + considered_radius, self.config["image_height"] - 1),
            min(x + considered_radius, self.config["image_height"] - 1))
        
        # Denormalize
        considered_patch = (255 * image[channel, boundary[0]:boundary[2], boundary[1]:boundary[3]]).astype(np.uint8)
        
        # Apply gaussian filter
        blurred_patch = cv.GaussianBlur(considered_patch, (kernel_size, kernel_size), 0)

        # Normalize
        blurred_patch = blurred_patch.astype(float) / 255

        # Overlay blurred patch to original image
        image[channel, boundary[0]:boundary[2], boundary[1]:boundary[3]] = blurred_patch

        return image


class LocalGaussianBlurringDefensePolicy(nn.Module):

    def __init__(self):
        super(LocalGaussianBlurringDefensePolicy, self).__init__()

        self.channel = nn.Linear(128, 3) # discrete action : select channel
        self.index = nn.Linear(128, 2)   # continuous action1 : select index num
        self.noise = nn.Linear(128, 2)   # continuous action2 : select noise std

        self.action_num = 3

    def policy(self, x, batch=False):
        channel_out = self.channel(x)
        index_out = self.channel(x)
        noise_out = self.channel(x)

        if batch:
            index_out = index_out.transpose(1, 0)
            noise_out = noise_out.transpose(1, 0)

        return (channel_out, index_out, noise_out)
    
    def get_actions(self, x, softmax_dim=0):
        channel_out, index_out, noise_out = x

        prob_channel = F.softmax(channel_out, dim=softmax_dim)
        dist_channel = Categorical(prob_channel)

        mu_index = torch.sigmoid(index_out[0])
        std_index = F.softplus(index_out[1])
        dist_index = Normal(mu_index, std_index)

        mu_noise = torch.sigmoid(noise_out[0])
        std_noise = F.softplus(noise_out[1])
        dist_noise = Normal(mu_noise, std_noise)


        a_channel = dist_channel.sample()
        log_prob_channel = dist_channel.log_prob(a_channel)

        a_index = dist_index.sample()
        log_prob_index = dist_index.log_prob(a_index)
        a_index = torch.clamp(a_index*1023, 0, 1023).int()

        a_noise = dist_noise.sample()
        log_prob_noise = dist_noise.log_prob(a_noise)
        a_noise = torch.clamp(a_noise*0.25, 0, 0.25).int()

        return (a_channel, a_index, a_noise), (log_prob_channel, log_prob_index, log_prob_noise)