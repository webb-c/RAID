import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Normal, Categorical
from utils import print_nested_info, load_model

import random


class ReplayBuffer:
    def __init__(self, buffer_size, minibatch_size):
        self.buffer = []
        self.batch = []
        if buffer_size > 1000:
            self.buffer_size = buffer_size
        else:
            print("Warning: DQN buffer size should be greater than 1000. Default buffer size set to 1000.")
            self.buffer_size = 1000
        self.minibatch_size = minibatch_size


    def _make_batch(self):
        """ object : buffer에 저장된 buffer_size * minibatch_size개의 데이터를 buffer_size개의 minibatch가 담긴 데이터로 전환하여 저장합니다."""
        sampled_data = random.sample(self.buffer, self.minibatch_size)

        a_batch, r_batch, done_batch = [], [], []
        img_batch, feat_batch, img_prime_batch, feat_prime_batch = [], [], [], []

        for transition in sampled_data:
            s, a, r, s_prime, done = transition
            img_batch.append(s[0])
            feat_batch.append(s[1])
            a_batch.append(a)
            r_batch.append([r])
            img_prime_batch.append(s_prime[0])
            feat_prime_batch.append(s_prime[1])
            done_mask = 0 if done else 1
            done_batch.append(done_mask)
            
        img_batch = torch.squeeze(torch.tensor(img_batch), 1)
        feat_batch = torch.squeeze(torch.tensor(feat_batch), 1)
        img_prime_batch = torch.squeeze(torch.tensor(img_prime_batch), 1)
        feat_prime_batch = torch.squeeze(torch.tensor(feat_prime_batch), 1)

        self.batch = [img_batch, feat_batch],  torch.tensor(a_batch, dtype=torch.float), torch.tensor(r_batch, dtype=torch.float), \
            [img_batch, feat_batch], torch.tensor(done_batch, dtype=torch.float)

    def clear(self):
        """ object: buffer를 비웁니다.
        input: None
        output: None
        """
        del self.buffer[:]
    
    
    def put(self, transition):
        """ object: buffer에 transition을 넣습니다.
        input: transition -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int, float], float, 
                Tuple[torch.Tensor, torch.Tensor], Tuple[float, float, float], bool]
        output: None
        """
        self.buffer.append(transition)
        self.buffer = self.buffer[-self.buffer_size:]
    
    
    def get_batch(self):
        """ object: 내부에서 _make_batch()를 호출하여 만든 batch data를 반환합니다.
        input: None
        output: self.batch_data -> List[[[List[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, 
                                        List[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], ... (#32)], ... (#10)]
        """
        self._make_batch()
        
        return self.batch
        
        
    def is_train_available(self):
        """ object: 현재 buffer가 minibatch를 만들 수 있을 정도로 가득 찬 상태인지 확인하고 그 결과를 반환합니다.
        input: None
        output: bool
        """
        if len(self.buffer) >= self.buffer_size * 0.1 :
            return True
        else:
            return False
        
class DQN(nn.Module):
    def __init__(self, config, policy):
        super(DQN, self).__init__()
        self.buffer = ReplayBuffer(config["buffer_size"], config["minibatch_size"])
        # parameter setting
        ### parameter for PPO
        self.mode = config["mode"]
        self.max_ep_len = config["max_ep_len"]
        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]          
        self.minibatch_size = config["minibatch_size"]   # M
        self.epsilon = 0.9

        ### parameter for RAID
        self.action_num = 1
        self.layer_idx = config["layer_idx"]
        self.alpha = config["alpha"]
        self.model_name = config["model_name"]

        # PPO model setting   
        self.backbone_part1, self.backbone_part2 = self._split_model()
        self.shared_layer = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # self.action = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 11),
        # )

        self.policy_network = policy()
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimization_step = 0

        # parameter for cuda
        self.available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def _split_model(self):
        """ object: backbone으로 사용할 모델을 로드하고 layer_idx를 기준으로 2개의 부분으로 모델을 나누어 반환합니다."""
        backbone = load_model(self.model_name)
        current_idx = 0
        part1_layers, part2_layers = [], []
        
        def _recursive_add_layers(module):
            nonlocal current_idx
            for child in module.children():
                if len(list(child.children())) == 0 : 
                    if current_idx <= self.layer_idx:
                        part1_layers.append(child)
                    else:
                        part2_layers.append(child)
                    current_idx += 1
                else: 
                    _recursive_add_layers(child)            
            return

        _recursive_add_layers(backbone)
        part1_model, part2_model = nn.Sequential(*part1_layers), nn.Sequential(*part2_layers[:-1])
        
        return part1_model, part2_model

    def _backbone(self, img, feature):
        """ object: img를 part1에 통과시키고, 그 결과를 feature와 addition한 뒤 part2에 통과시켜 얻은 최종 feature를 반환합니다."""
        mid_feature = self.backbone_part1(img)
        agent_feature = self.backbone_part2(mid_feature - feature)
        agent_feature = torch.squeeze(agent_feature)
        
        return agent_feature
    
        
    def _policy(self, agent_feature, softmax_dim=0, batch=False) -> torch.Tensor:
        """ object: backbone을 통과하여 얻어진 feature를 각각의 policy layer를 통과시켜 action의 distribution을 계산해 반환합니다."""
        x = self.shared_layer(agent_feature)
        
        x = self.policy_network.policy(x)
        
        return x
    
    def get_q_table(self, state, train=False):
        """ object: input을 받아, 내부에서 _backbone, _policy 함수를 호출한 뒤 action과 해당 action의 log_prob들을 tuple로 반환합니다.
        *만약 train하는 과정이라면 batch 단위로 실행되기 때문에 output이 (32, .) 형태의 Tensor로 반환됩니다.
        input: state -> Tuple[torch.Tensor, torch.Tensor]; train -> bool
        output: actions -> Tuple[int, int, float]; probs -> Tuple[float, float, float]
        """
        img, feature = state
        if not train:
            self.eval()
            img = torch.tensor(img).unsqueeze(0)
            feature = torch.tensor(feature)
            with torch.no_grad():
                agent_feature = self._backbone(img, feature)
                q_table = self._policy(agent_feature, softmax_dim=0)
        else :
            img = img.to(self.available_device)
            feature = feature.to(self.available_device)
            agent_feature = self._backbone(img, feature)
            q_table = self._policy(agent_feature, softmax_dim=1, batch=True)
     
        return q_table
    
    
    def get_actions(self, state, train=False, rand=False):
        """ object: input을 받아, 내부에서 _backbone, _policy 함수를 호출한 뒤 action과 해당 action의 log_prob들을 tuple로 반환합니다.
        *만약 train하는 과정이라면 batch 단위로 실행되기 때문에 output이 (32, .) 형태의 Tensor로 반환됩니다.
        input: state -> Tuple[torch.Tensor, torch.Tensor]; train -> bool
        output: actions -> Tuple[int, int, float]; probs -> Tuple[float, float, float]
        """
        q_table = self.get_q_table(state, train=train)
        coin = random.random()
        if coin < self.epsilon or rand:
            return (random.randint(0, q_table.shape[0]), )
        else : 
            return (q_table.argmax().item(), )


    def put_data(self, transition):
        """ object: 1번의 transition 데이터를 Tensor타입을 제거한 뒤 put()을 호출하여 buffer에 집어넣습니다.
        input: transition -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int, float], float, 
                                Tuple[torch.Tensor, torch.Tensor], Tuple[float, float, float], bool]
        output: None
        """
        s, a, r, s_prime, done = transition
        s_noTensor = [data.tolist() for data in s]
        s_prime_noTensor = [data.tolist() for data in s_prime]
        self.buffer.put((s_noTensor, a, r, s_prime_noTensor, done))

        
    def train_net(self, q_target):
        """ object: buffer가 가득차면 buffer에 쌓여있는 데이터를 사용하여 K_epochs번 DNN의 업데이트를 진행합니다.
        input: None
        output: None or loss
        """
        self.epsilon = max(0.05, self.epsilon - 0.000003)
        q_target.to(self.available_device)
        total_loss = None
        if self.buffer.is_train_available():
            self = self.to(self.available_device)
            total_loss = 0
            
            for i in range(5):
                batch = self.buffer.get_batch()
                s, a, r, s_prime, done_mask = batch
                
                a = a.to(self.available_device).long().view(-1, 1)
                r = r.to(self.available_device).view(-1, 1)
                done_mask = done_mask.to(self.available_device).view(-1, 1)

                q_out = self.get_q_table(s, train=True)
                q_a = torch.gather(q_out, 1, a)
                
                max_q_prime = q_target.get_q_table(s_prime, train=True).max(1)[0].view(-1, 1)

                target = r + self.gamma * max_q_prime * done_mask
                
                loss = F.smooth_l1_loss(q_a, target)
                total_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # episode 시작하기 위해서 cpu로 보냄
            self.cpu()
        
        return total_loss
                    