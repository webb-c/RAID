from defense.DefenseBase import DefenseBase
import numpy as np
import cv2 as cv
from preprocess.RotatePreprocess import RotatePreprocess
from preprocess.BrightnessControlPreprocess import BrightnessControlPreprocess
from preprocess.VerticalFlipPreprocess import VerticalFlipPreprocess
from preprocess.HorizontalFlipPreprocess import HorizontalFlipPreprocess
from preprocess.VerticalShiftPreprocess import VerticalShiftPreprocess
from preprocess.HorizontalShiftPreprocess import HorizontalShiftPreprocess
from preprocess.ResizePreprocess import ResizePreprocess

class MultiAugmentation(DefenseBase):

    def __init__(self, config) -> None:
        self.config = config
        self.rotate_preprocess = RotatePreprocess()
        self.brightness_control_preprocess = BrightnessControlPreprocess()
        self.vertical_flip_preprocess = VerticalFlipPreprocess()
        self.horizontal_flip_preprocess = HorizontalFlipPreprocess()
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

        match preprocessing_index:
            case 0:
                preprocessed_image = self.rotate_preprocess.process(image, 10)
            case 1:
                preprocessed_image = self.rotate_preprocess.process(image, -10)
            case 2:
                preprocessed_image = self.brightness_control_preprocess.process(image, 0.05)
            case 3:
                preprocessed_image = self.brightness_control_preprocess.process(image, -0.05)
            case 4:
                preprocessed_image = self.vertical_flip_preprocess.process(image)
            case 5:
                preprocessed_image = self.horizontal_flip_preprocess.process(image)
            case 6:
                preprocessed_image = self.vertical_shift_preprocess.process(image, 0.1)
            case 7:
                preprocessed_image = self.vertical_shift_preprocess.process(image, -0.1)
            case 8:
                preprocessed_image = self.horizontal_shift_preprocess.process(image, 0.1)
            case 9:
                preprocessed_image = self.horizontal_shift_preprocess.process(image, -0.1)
            case 10:
                preprocessed_image = self.resize_preprocess.process(image, 0.9)


        return preprocessed_image


