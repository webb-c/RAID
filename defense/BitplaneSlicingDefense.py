import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from defense.DefenseBase import DefenseBase

from torch.distributions import Categorical, Normal

class BitplaneSlicingDefense(DefenseBase):

    def __init__(self, config) -> None:
        self.config = config
    
    def apply(self, image, action):
        """
        BitplaneSlicingDefense는 주어진 target bit의 값을 0으로 변환합니다.

        input:
            - image (np.ndarray) : 전처리할 이미지
            - action (tuple) : 적용할 target bit
        """
        index = action[0].item()

        # Denormalize
        image = (255 * image).astype(np.uint8)

        # Apply bitwise slicing calculation
        image = np.bitwise_and(image, ~(1 << index))

        # Normalize
        image = image.astype(np.float32) / 255

        return image


class BitplaneSlicingDefensePolicy(nn.Module):

    def __init__(self):
        super(BitplaneSlicingDefensePolicy, self).__init__()

        self.index = nn.Linear(128, 8) # discrete action : select index

        self.action_num = 1

    def policy(self, x, batch=False):
        index_out = self.index(x)

        return (index_out, )
    
    def get_actions(self, x, softmax_dim=0, rand=False):
        if rand :
            return (random.randint(0, 7), ), (0, ), (0, )
        
        index_out = x[0]

        prob_index = F.softmax(index_out, dim=softmax_dim)
        dist_index = Categorical(prob_index)

        a_index = dist_index.sample()
        log_prob_index = dist_index.log_prob(a_index)

        return (a_index,), (log_prob_index,), (dist_index.entropy(),)