from preprocess.PreprocessBase import PreprocessBase
import cv2 as cv

class HorizontalFlipPreprocess(PreprocessBase):

    def process(self, image, *args, **kwargs):
        """"
        VerticalFlipPreprocess는 이미지를 수평 방향으로 뒤집습니다.

        input:
            - image (np.ndarray): 변형할 이미지
        """

        image = image.transpose(1, 2, 0)

        preprocessed_image = cv.flip(image, 1)

        preprocessed_image = preprocessed_image.transpose(2, 0, 1)

        return preprocessed_image