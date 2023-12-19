from defense.DefenseBase import DefenseBase
import numpy as np
import random
import cv2 as cv
from preprocess.RotatePreprocess import RotatePreprocess
from preprocess.BrightnessControlPreprocess import BrightnessControlPreprocess
from preprocess.VerticalFlipPreprocess import VerticalFlipPreprocess
from preprocess.HorizontalFlipPreprocess import HorizontalFlipPreprocess
from preprocess.VerticalShiftPreprocess import VerticalShiftPreprocess
from preprocess.HorizontalShiftPreprocess import HorizontalShiftPreprocess
from preprocess.ResizePreprocess import ResizePreprocess

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class MultiAugmentationShort(DefenseBase):

    def __init__(self, config) -> None:
        self.config = config
        self.rotate_preprocess = RotatePreprocess()
        self.vertical_shift_preprocess = VerticalShiftPreprocess()
        self.horizontal_shift_preprocess = HorizontalShiftPreprocess()
        self.resize_preprocess = ResizePreprocess()
    
    def apply(self, image, action):
        """"
        MultiAugmentation는 여러가지의 augmentation process를 적용합니다.

        input:
            - image (np.ndarray) : 전처리할 이미지
            - action (List[int]) : 적용할 전처리 종류
        """

        preprocessing_index, = action

        if preprocessing_index == 0:
            preprocessed_image = self.rotate_preprocess.process(image, 5)
        elif preprocessing_index == 1:
            preprocessed_image = self.rotate_preprocess.process(image, -5)
        elif preprocessing_index == 2:
            preprocessed_image = self.vertical_shift_preprocess.process(image, 0.05)
        elif preprocessing_index == 3:
            preprocessed_image = self.vertical_shift_preprocess.process(image, -0.05)
        elif preprocessing_index == 4:
            preprocessed_image = self.horizontal_shift_preprocess.process(image, 0.05)
        elif preprocessing_index == 5:
            preprocessed_image = self.horizontal_shift_preprocess.process(image, -0.05)
        elif preprocessing_index == 6:
            preprocessed_image = self.resize_preprocess.process(image, 0.95)


        return preprocessed_image


class MultiAugmentationShortPolicy(nn.Module):

    def __init__(self):
        super(MultiAugmentationShortPolicy, self).__init__()

        self.index = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

        self.action_num = 1

    def policy(self, x, batch=False):
        index_out = self.index(x)
        return index_out
    
    def get_actions(self, x, softmax_dim=0, rand=False):
        if rand:
            return (random.randint(0, 6), ), (0, ), (0, )
        index_out = x

        prob_index = F.softmax(index_out, dim=softmax_dim)

        dist_index = Categorical(prob_index)

        a_index = dist_index.sample()
        log_prob_index = dist_index.log_prob(a_index)

        return (a_index, ), (log_prob_index, ), (dist_index.entropy(), )

