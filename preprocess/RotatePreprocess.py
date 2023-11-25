from preprocess.PreprocessBase import PreprocessBase
import cv2 as cv

class RotatePreprocess(PreprocessBase):

    def process(self, image, angle = 10, *args, **kwargs):
        """"
        RotatePreprocess는 주어진 angle만큼 이미지를 회전시킵니다.

        input:
            - image (np.ndarray): 변형할 이미지
            - angle (int): 변형할 각도 값, 정수 단위
        """

        image = image.transpose(1, 2, 0)

        H, W, C = image.shape

        M = cv.getRotationMatrix2D((H // 2, W // 2), 2, 1)
        preprocessed_image = cv.warpAffine(image, M, (H, W))

        preprocessed_image = preprocessed_image.transpose(2, 0, 1)

        return preprocessed_image
