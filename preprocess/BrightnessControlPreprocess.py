from preprocess.PreprocessBase import PreprocessBase
import numpy as np

class BrightnessControlPreprocess(PreprocessBase):

    def process(self, image, brightness = 0.1, *args, **kwargs):
        """"
        BrightnessControlPreprocess는 주어진 brightness만큼 이미지의 밝기를 높이거나 낮춥니다.

        input:
            - image (np.ndarray): 변형할 이미지
            - brightness (float): 변형할 brightness 값, 0 ~ 1
        """
        
        preprocessed_image = image + brightness
        preprocessed_image = np.clip(preprocessed_image, 0, 1)

        return preprocessed_image