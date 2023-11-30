import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from defense.DefenseBase import DefenseBase
from utils import get_filtered_image


class HighFrequencyDrop(DefenseBase):
    def __init__(self, config) -> None:
        self.config = config
    
    def apply(self, image, action):
        """
        HigeFrequencyDropDefense는 Frequency Domain에서 radius를 갖는 원을 기준으로 원 내부에 있는 픽셀만을 취한뒤 다시 도메일을 변환하여 이미지로 반환합니다..
            - image (np.ndarray) : 전처리할 이미지
            - action (List[int]) : 반지름
        """
        preprocessed_image = get_filtered_image(image, r=action)
        
        return preprocessed_image


class HighFrequencyDropPolicy(nn.Module):

    def __init__(self):
        super(HighFrequencyDropPolicy, self).__init__()

        self.index = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 17),   # [0, 16]
        )
        self.action_num = 1


    def policy(self, x, batch=False):
        index_out = self.index(x)
        return index_out
    
    
    def get_actions(self, x, softmax_dim=0):
        index_out = x
        prob_index = F.softmax(index_out, dim=softmax_dim)

        dist_index = Categorical(prob_index)

        a_index = dist_index.sample()
        log_prob_index = dist_index.log_prob(a_index)

        return (a_index, ), (log_prob_index, )
