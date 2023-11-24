from abc import ABC, abstractmethod
import numpy as np
 
class PreprocessBase(ABC):

    @abstractmethod
    def process(self, image, *args, **kwargs):
        """_summary_

        Args:
            image (np.ndarray): Envrionment에서 받은 현재 이미지
            action (List): actions

        Returns:
            np.ndarray: preprocessed image
        """
        pass

    def tensor2uint8(self, image):
        return (255 * image).astype(np.uint8)
    
    def uint82tensor(self, image):
        return (image / 255).astype(np.float32)

    def __call__(self, image, *args, **kwargs) -> None:
        return self.process(image, *args, **kwargs)
    