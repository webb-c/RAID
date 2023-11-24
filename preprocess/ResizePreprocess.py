from preprocess.PreprocessBase import PreprocessBase
import numpy as np
import cv2 as cv

class ResizePreprocess(PreprocessBase):

    def process(self, image, ratio = 0.9):
        """"
        ResizePreprocess는 주어진 ratio만큼 이미지의 크기를 변경하고, 빈 부분을 검은색으로 채웁니다.

        input:
            - image (np.ndarray): 변형할 이미지, (C, H, W)
            - ratio (float): 현재 크기에서 ratio만큼 이미지를 줄이거나 크게 만듭니다.
        """

        assert ratio > 0, "Resizing ratio should be greater than 0."

        image = image.transpose(1, 2, 0)

        H, W, C = image.shape

        # selece interpolation method by ratio
        interpolation_method = cv.INTER_CUBIC if ratio > 1.0 else cv.INTER_AREA

        # Calculate the new size of the image
        new_size = (int(H * ratio), int(W * ratio))

        # Resize the image
        resized_image = cv.resize(image, new_size, interpolation=interpolation_method)

        # Create a new image with the target size and fill with black color
        preprocessed_image = np.zeros_like(image, dtype=np.float32)

        # Calculate the position to place the resized image in the new image
        x_offset = (H - new_size[0]) // 2
        y_offset = (W - new_size[1]) // 2

        # Place the resized image in the center of the new image
        preprocessed_image[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized_image

        preprocessed_image = preprocessed_image.transpose(2, 0, 1)

        return preprocessed_image
    

