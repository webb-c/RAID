from defense.DefenseBase import DefenseBase
import numpy as np
import cv2 as cv

class LocalGaussianBlurringDefense(DefenseBase):

    def __init__(self, config) -> None:
        self.config = config
    
    def apply(self, image, action):
        """
        LocalGaussianBlurringDefense는 주어진 target pixel(index) 주위에 std만큼의 gaussian blur를 적용합니다.

        input:
            - channel (int): 변형할 이미지의 channel (0: R, 1: G, 2: B)
            - index (int): 변형할 image의 index vector
            - std (float): defense perturbation의 standard deviation
        """
        channel, index, std = action

        y = index // self.config["image_height"]
        x = index % self.config["image_width"]
        kernel_radius = int(self.config["image_height"] * std) + 1
        kernel_size = kernel_radius * 2 + 1
        considered_radius = kernel_radius * 2

        # Patch coordinate with form (y1, x1, y2, x2)
        boundary = (
            max(y - considered_radius, 0),
            max(x - considered_radius, 0),
            min(y + considered_radius, self.config["image_height"] - 1),
            min(x + considered_radius, self.config["image_height"] - 1))
        
        # Denormalize
        considered_patch = (255 * image[channel, boundary[0]:boundary[2], boundary[1]:boundary[3]]).astype(np.uint8)
        
        # Apply gaussian filter
        blurred_patch = cv.GaussianBlur(considered_patch, (kernel_size, kernel_size), 0)

        # Normalize
        blurred_patch = blurred_patch.astype(float) / 255

        # Overlay blurred patch to original image
        image[channel, boundary[0]:boundary[2], boundary[1]:boundary[3]] = blurred_patch

        return image


