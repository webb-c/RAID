from preprocess.PreprocessBase import PreprocessBase
import numpy as np

class HorizontalShiftPreprocess(PreprocessBase):

    def process(self, image, ratio = 0.1, *args, **kwargs):
        """"
        HorizontalShiftPreprocessing는 주어진 ratio만큼 이미지를 수평 방향으로 움직입니다.

        input:
            - image (np.ndarray): 변형할 이미지
            - ratio (float): 움직일 비율
        """

        assert -1 < ratio < 1, "Ratio should be greater than -1 and less than 1"

        image = image.transpose(1, 2, 0)

        H, W, C = image.shape

        to_shift = H * abs(ratio)

        # shift right
        if ratio > 0:
            preprocessed_image = np.zeros_like(image)
            preprocessed_image[:, int(to_shift):, :] = image[:, :int(W - to_shift) + 1, :]
        # shift up
        elif ratio < 0:
            preprocessed_image = np.zeros_like(image)
            preprocessed_image[:, :int(W - to_shift) + 1, :] = image[:, int(to_shift):, :]

        preprocessed_image = preprocessed_image.transpose(2, 0, 1)

        return preprocessed_image
  

