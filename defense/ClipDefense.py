from defense.DefenseBase import DefenseBase
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal

class ClipDefense(DefenseBase):
    # for discrete actions - percentage
    def __init__(self, config) -> None:
        self.config = config
    
    def apply(self, image, action):
        """
        ClipDefense는 0과 1 사이인 이미지 값을 clip_center 중심으로 clip_len 길이만큼 clamp합니다.

        input:
            - clip_center (int): clip할 범위의 중간값, 0, 1, 2, 3, 4 -> 0.0, 0.25, 0.5, 0.75, 1.0
            - clip_len (int): clip할 percentage, 0, 1, 2, 3 -> 0.25, 0.5, 0.75, 1.0
        result:
            - [clip_center-clip_len/2, clip_center+clip_len/2] 범위로 clip된 이미지
        """
        center, ratio = action

        # current image information
        cur_min = np.min(image)
        cur_max = np.max(image)
        cur_len = cur_max - cur_min

        if center == 0:
            clip_center = 0.0
        elif center == 1:
            clip_center = 0.25
        elif center == 2:
            clip_center = 0.5
        elif center == 3:
            clip_center = 0.75
        elif center == 4:
            clip_center = 1.0
        
        if ratio == 0:
            clip_ratio = 0.25
        elif ratio == 1:
            clip_ratio = 0.5
        elif ratio == 2:
            clip_ratio = 0.75
        elif ratio == 3:
            clip_ratio = 1.0
        
        clip_len = cur_len * clip_ratio
        center_len = cur_len - clip_len
        clip_center = cur_min + clip_len/2 + center_len * clip_center

        clip_min = max(clip_center-clip_len/2, 0.0)
        clip_max = min(clip_center+clip_len/2, 1.0)

        image = np.clip(image, a_min=clip_min, a_max=clip_max)

        return image
    
class ClipDefensePolicy(nn.Module):
    # discrete mode
    def __init__(self):
        super(ClipDefensePolicy, self).__init__()

        self.center = nn.Linear(128, 5)     # discrete action1 : select center of clip range [0.0, 0.25, 0.5, 0.75, 1.0]
        self.length = nn.Linear(128, 4)     # discrete action2 : select percentage of clip range [0.25, 0.5, 0.75, 1.0]
        
        self.action_num = 2

    def policy(self, x, batch=False):
        center_out = self.center(x)
        length_out = self.length(x)

        return (center_out, length_out)
    
    def get_actions(self, x, softmax_dim=0, rand=False):
        if rand:
            return (random.randint(0, 4), random.randint(0, 3)), (0, 0), (0, 0)
        
        center_out, length_out = x
        
        prob_center = F.softmax(center_out, dim=softmax_dim)
        dist_center = Categorical(prob_center)

        prob_length = F.softmax(length_out, dim=softmax_dim)
        dist_length = Categorical(prob_length)

        a_center = dist_center.sample()
        log_prob_center = dist_center.log_prob(a_center)

        a_length = dist_length.sample()
        log_prob_length = dist_length.log_prob(a_length)

        return (a_center, a_length), (log_prob_center, log_prob_length), (dist_center.entropy(), dist_length.entropy())