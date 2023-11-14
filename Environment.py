import argparse
import torch
import glob
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from typing import Union, Generator, Tuple
from utils import register_single_hook, return_feature_map, load_model, hook_fn



class Env():
    def __init__(self, model_name: str = "mobilenet", mode: str = "train", feature_extract_layer: int = 4) -> None:
        self.state: Tuple[np.ndarray, np.ndarray] = None # current state (image, featuremap)
        self.target_label: int = None
        self.target_image: np.ndarray = None
        self.episode: int = None # current episode (integer)
        self.prev_confidence_score: float = None

        self.model_name: str = model_name

        self.permutation_list: np.ndarray = None

        self.train_dataset: dict = None
        self.val_dataset: dict = None
        self.test_dataset: dict = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])

        self.mode = mode
        self.feature_extract_layer = feature_extract_layer

        self.image_model: torch.nn.Module = self._load_model()
        self._load_dataset()

    # return permutation list
    def _make_permutation_list(self, n) -> np.ndarray:
        """ 
        _make_permutation_list 함수는 [0,n-1] 에 속하는 정수로 permutation vector를 반환합니다.
        
        input 
            n : 만들고 싶은 permutation vector의 길이
        output
            없음 
        """
        self.permutation_list = np.random.permutation(n)

    def _load_dataset(self) -> None:
        """ 
        _load_dataset 함수는 original image와 perturbed 이미지를 불러옵니다.
        self.mode에 따라서 train, val, test으로 서로 다른 데이터셋을 self.train_dataset, self.val_dataset, self.test_dataset 에 로드합니다.
        """
        origin_images_paths = glob.glob(f"images/{self.model_name}/origin/{self.mode}")
        perturbed_images_paths = glob.glob(f"images/{self.model_name}/adv/{self.mode}")

        data_num = len(origin_images_paths)

        self._make_permutation_list(n = data_num)
        
        origin_images = (self._return_transform_image(path) for path in origin_images_paths)
        perturbed_images = (self._return_transform_image(path) for path in perturbed_images_paths)

        if self.mode == "train":
            self.train_dataset = {"origin_images" : origin_images, "perturbed_images" : perturbed_images}

        elif self.mode == "val":
            self.val_dataset = {"origin_images" : origin_images, "perturbed_images" : perturbed_images}

        elif self.mode == "test":
            self.test_dataset = {"origin_images" : origin_images, "perturbed_images" : perturbed_images}


    def _return_transform_image(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as img:
            return self.transform(img)
        
    def _load_model(self) -> torch.nn.Module:
        model_path = f"models/{self.model_name}.pt"
        image_model = load_model(self.model_name, model_path)
        register_single_hook(image_model, self.feature_extract_layer, hook_fn)

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
        
        if self.mode == "train":
            dataset = self.train_dataset

        elif self.mode == "val":
            dataset = self.val_dataset

        elif self.mode == "test":
            dataset = self.test_dataset
            
        origin_images = dataset["origin_images"]
        perturbed_images = dataset["perturbed_images"]

        origin_image_path = next(origin_images, None)
        perturbed_image_path = next(perturbed_images, None)

        if origin_image_path is None and perturbed_image_path is None:
            return -1
        
        origin_image_tensor: torch.Tensor = self.transform(origin_image_path)
        perturbed_image_tensor: torch.Tensor = self.transform(perturbed_image_path)

        image_label: int = int(perturbed_image_path.split("_")[1])

        if image_label != int(origin_image_path.split("_")[1]):
            raise ValueError("The original class of the original image and the perturbed image is different.")

        return (origin_image_tensor, perturbed_image_tensor, image_label)
        
    

    def _inference(self, image: torch.Tensor = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        _inference 함수는 입력에 대한 target DNN 모델의 confidence score와 중간 feature를 반환합니다.

        input:
            image: 추론을 진행할 torch.Tensor 이미지
        output:
            (confidence score, feature map)
        """
        with torch.no_grad():
            confidence_score = np.array(torch.nn.functional.softmax(self.image_model(image), dim=0))
        feature_map = np.array(return_feature_map(self.image_model))

        return (confidence_score, feature_map)
                

    def train(self) -> None:
        self.mode = "train"

        if self.train_dataset is None:
            self._load_dataset()

    def val(self) -> None:
        self.mode = "val"

        if self.val_dataset is None:
            self._load_dataset()

    def test(self) -> None:
        self.mode = "test"

        if self.test_dataset is None:
            self._load_dataset()
        

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        reset 함수는 state, episode, prev_confidence_score, dataset을 초기화 하고, state를 반환합니다.
        self.mode에 따라 state가 불러와지는 dataset이 달라집니다.

        input:
            없음
        output:
            (image: np.ndarray, feature_map: np.ndarray)
        """
        origin_image_tensor, perturbed_image_tensor, image_label = self._get_next_image_label()
        confidence_score, feature_map = self._inference()

        self.episode += 1
        self.target_image = np.array(origin_image_tensor)
        self.target_label = image_label
        self.state = (np.array(perturbed_image_tensor), np.array(feature_map))

        self.prev_confidence_score = confidence_score
        
        return self.state
