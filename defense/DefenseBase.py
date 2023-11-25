from abc import ABC, abstractmethod
 
class DefenseBase(ABC):

    @abstractmethod
    def apply(self, *args, **kwargs):
        """_summary_

        Args:
            image (np.ndarray): Envrionment에서 받은 현재 이미지
            action (List): actions

        Returns:
            np.ndarray: preprocessed image
        """
        pass

    def __call__(self, *args, **kwargs) -> None:
        return self.apply(*args, **kwargs)