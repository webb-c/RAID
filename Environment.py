import cv2 as cv
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms

import glob
from typing import Union, Tuple

from utils import register_single_hook, return_feature_map, load_model, hook_fn

class Env():
    def __init__(self,
                 args: dict = {'learning_rate': 0.0003, 'gamma': 0.9, 'lmbda': 0.9, 'alpha': 0.5, 'eps_clip': 0.2, 'num_epoch': 10, 'num_step': 50, 'rollout_len': 3, 'buffer_size': 10, 'minibatch_size': 32, 'mode': 'train', 'model_name': 'mobilenet', 'dataset_name': 'CIFAR10', 'layer_idx': 4}
                 ) -> None:
        
        self.state: list[np.ndarray] = None # current state [image, featuremap]
        self.target_label: int = None
        self.target_image: np.ndarray = None
        self.episode: int = 0 # current episode (integer)
        self.prev_confidence_score: np.ndarray = None
        self.alpha: float = args["alpha"]
        self.size: int = 32
        self.epoch: int = 1
        self.truncated_step = 10 # TODO get value from args

        self.model_name: str = args["model_name"]

        self.dataset: np.ndarray = None

        self.train_dataset: dict = None
        self.val_dataset: dict = None
        self.test_dataset: dict = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])

        self.mode = args["mode"]
        self.layer_idx = args["layer_idx"]

        self.image_model: torch.nn.Module = self._load_model()
        self._load_dataset()

    # return permutation list
    def _get_permutation_list(self, n) -> np.ndarray:
        """ 
        _make_permutation_list 함수는 [0,n-1] 에 속하는 정수로 permutation vector를 반환합니다.
        
        input 
            n : 만들고 싶은 permutation vector의 길이
        output
            permutation_vector : [0, n-1] 가 중복되지 않고 섞여있는 np.ndarray
        """
        permutation_vector = np.random.permutation(n)
        return permutation_vector

    def _load_dataset(self) -> None:
        """ 
        _load_dataset 함수는 self.mode에 맞는 original image, perturbed image, label을 불러와 [self.train_dataset, self.val_dataset, self.test_dataset] 중 self.mode에 해당하는 변수에 저장합니다.
        """
        original_images_paths = glob.glob(f"images/{self.model_name}/origin/{self.mode}/*")
        perturbed_images_paths = glob.glob(f"images/{self.model_name}/adv/{self.mode}/*")


        original_images = original_images_paths
        perturbed_images = perturbed_images_paths
        original_classes = [self._get_class(image_path) for image_path in original_images_paths]

        if self.mode == "train":
            self.train_dataset = {"original_images" : original_images, "perturbed_images" : perturbed_images, "original_classes" : original_classes, "num_images" : len(original_images)}

        elif self.mode == "val":
            self.val_dataset = {"original_images" : original_images, "perturbed_images" : perturbed_images, "original_classes" : original_classes, "num_images" : len(original_images)}

        elif self.mode == "test":
            self.test_dataset = {"original_images" : original_images, "perturbed_images" : perturbed_images, "original_classes" : original_classes, "num_images" : len(original_images)}


    def _set_dataset(self) -> None:

        if self.mode == "train":
            dataset = self.train_dataset

        elif self.mode == "val":
            dataset = self.val_dataset

        elif self.mode == "test":
            dataset = self.test_dataset

        data_num = dataset["num_images"]

        permutation_list = self._get_permutation_list(n = data_num)
        
        original_images = (self._get_transform_image(dataset["original_images"][index]) for index in permutation_list)
        perturbed_images = (self._get_transform_image(dataset["perturbed_images"][index]) for index in permutation_list)
        original_classes = (dataset["original_classes"][index] for index in permutation_list)

        self.dataset = {"original_images" : original_images, "perturbed_images" : perturbed_images, "original_classes" : original_classes}

        if len(permutation_list) > 0:
            print(f"Current mode : {self.mode}")
            print(f"{len(permutation_list)} images succesfully loaded")
        else:
            print(f"error")


    def _get_transform_image(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as img:
            return self.transform(img)
        
    def _get_class(self, image_name: str) -> int:
        return int(image_name.split("_")[1].split(".")[0])
        
    def _load_model(self) -> torch.nn.Module:
        model_path = f"models/{self.model_name}.pt"
        image_model = load_model(self.model_name, model_path)
        register_single_hook(image_model, self.layer_idx, hook_fn)

        return image_model
    
    
    def _get_next_image_label(self) -> Union[Tuple[torch.Tensor, torch.Tensor, int], int]:
        """
        _get_next_image_label 함수는 현재 선택된 데이터셋에서 다음 이미지의 original, perturbed, class를 반환합니다. 
        self.mode에 따라서 self.train_dataset, self.val_dataset, self.test_dataset 에서 데이터를 가져옵니다.

        input:
            없음
        output:
            (origin_image_tensor, perturbed_image_tensor, image_label)
        """
            
        original_images = self.dataset["original_images"]
        perturbed_images = self.dataset["perturbed_images"]
        original_classes = self.dataset["original_classes"]

        origin_image = next(original_images, None)
        perturbed_image = next(perturbed_images, None)
        original_class = next(original_classes, None)

        if origin_image is None and perturbed_image is None and original_class is None:
            return -1
        
        return (origin_image, perturbed_image, original_class)
        
    

    def _inference(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        _inference 함수는 입력에 대한 target DNN 모델의 confidence score와 중간 feature를 반환합니다.

        input:
            image: 추론을 진행할 torch.Tensor 이미지
        output:
            (confidence score, feature map)
        """

        image = torch.tensor(self.state[0]).unsqueeze(0)

        with torch.no_grad():
            logit = self.image_model(image)

        prob = torch.nn.functional.softmax(logit, dim=1)
        confidence_score = np.array(prob)

        feature_map = np.array(return_feature_map(self.image_model, self.layer_idx))

        return (confidence_score, feature_map)

    def _defense_image(self, channel: int, index: int, std: float) -> None:
        """
        _defense_image 함수는 주어진 target pixel(index) 주위에 std만큼의 gaussian blur를 적용합니다.

        input:
            - channel (int): 변형할 이미지의 channel (0: R, 1: G, 2: B)
            - index (int): 변형할 image의 index vector
            - std (float): defense perturbation의 standard deviation
        """
        y = index / self.size
        x = index % self.size
        kernel_radius = int(self.size * std)
        kernel_size = kernel_radius * 2 + 1
        considered_radius = kernel_radius * 2

        # Patch coordinate with form (y1, x1, y2, x2)
        boundary = (
            max(y - considered_radius, 0),
            max(x - considered_radius, 0),
            min(y + considered_radius, self.size - 1),
            min(x + considered_radius, self.size - 1))
        
        # Denormalize
        considered_patch = (255 * self.state[0][channel, boundary[0]:boundary[2], boundary[1]:boundary[3]]).astype(np.uint8)

        # Apply gaussian filter
        blurred_patch = cv.GaussianBlur(considered_patch, (kernel_size, kernel_size), 0)

        # Normalize
        blurred_patch = blurred_patch.astype(float) / 255

        # Overlay blurred patch to original image
        self.state[0][channel, boundary[0], boundary[1]] = blurred_patch
        

    def _get_reward(self, confidence_score: np.ndarray) -> float:
        """
        _get_reward 함수는 이미지 모델의 추론에 대한 confidence drift의 정도를 반환합니다.

        input:
            - confidence_score (np.ndarray): 갱신된 confidence_score
        output:
            - reward (float): action을 수행했을 때의 reward를 반환합니다. Reward는 image model의 confidence drift입니다.
        """
        target_drift = confidence_score - self.prev_confidence_score
        non_target_drift = np.delete(confidence_score, self.target_label) - np.delete(self.prev_confidence_score, self.target_label)
        result = target_drift - self.alpha * non_target_drift

        return result
                

    def train(self) -> None:
        self.mode = "train"

        if self.train_dataset is None:
            self._load_dataset()
        self._set_dataset()

    def val(self) -> None:
        self.mode = "val"

        if self.val_dataset is None:
            self._load_dataset()
        self._set_dataset()

    def test(self) -> None:
        self.mode = "test"

        if self.test_dataset is None:
            self._load_dataset()
        self._set_dataset()

    def reset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
        """
        reset 함수는 state, episode, prev_confidence_score, dataset을 초기화 하고, state를 반환합니다.
        self.mode에 따라 state가 불러와지는 dataset이 달라집니다.

        input:
            없음
        output:
            (image: np.ndarray, feature_map: np.ndarray)
        """
        next = self._get_next_image_label()

        if next == -1:
            self.epoch += 1
            self._set_dataset()
            origin_image_tensor, perturbed_image_tensor, image_label = self._get_next_image_label()
        else:
            origin_image_tensor, perturbed_image_tensor, image_label = next

        self.state = [np.array(perturbed_image_tensor), None]
        confidence_score, feature_map = self._inference()

        self.episode += 1
        self.target_image = np.array(origin_image_tensor)
        self.target_label = image_label
        self.state = [np.array(perturbed_image_tensor), np.array(feature_map)]

        self.prev_confidence_score = confidence_score
        
        return (self.state, self.epoch)

    def step(self, action: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, bool, None]:
        """
        step 함수는 environment에 행할 action을 받아 해당 action을 진행했을 때의 state, reward, termination 여부를 반환합니다.

        input:
            - action (channel: int, index: int, std: float)
                - channel (int): 변형할 이미지의 channel (0: R, 1: G, 2: B)
                - index (int): 변형할 이미지의 pixel index
                - std (float) perturbation standard deviation
        output:
            - state (image: np.ndarray, feature_map: np.ndarray): action이 수행된 이후의 image array와 해당 array의 feature map을 반환합니다.
            - reward (float): action을 수행했을 때의 reward를 반환합니다. Reward는 image model의 confidence drift입니다.
            - terminated (bool): agent가 episode의 terminal state에 도착했는지의 여부입니다.
            - truncated (bool): agent가 episode 도중 truncation condition에 의해 중단되었는지의 여부입니다.
            - info (None): None을 반환합니다.
        """
        channel, index, std = action
        state = (np.zeros(1), np.zeros(1))
        terminated = False
        truncated = False

        # Defense image with action
        self._defense_image(channel, index, std)

        # Inference image, get new confidence score
        # Question: _inference가 인자를 가져야 하나?
        confidence_score, feature_map = self._inference()

        # Calculate confidence drift
        reward = self._get_reward(confidence_score)

        # Update attributes
        self.state[1] = feature_map
        self.prev_confidence_score = confidence_score

        return state, reward, terminated, truncated, None

