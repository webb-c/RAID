import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Normal, Categorical
from utils import print_nested_info, load_model


class RolloutBuffer:
    def __init__(self, buffer_size, minibatch_size):
        self.buffer = []
        self.batch_data = []
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size


    def _make_batch(self):
        """ object : buffer에 저장된 buffer_size * minibatch_size개의 데이터를 buffer_size개의 minibatch가 담긴 데이터로 전환하여 저장합니다."""
        self.batch_data = []
        for i in range(self.buffer_size):
            a_batch, r_batch, prob_a_batch, done_batch = [], [], [], []
            img_batch, feat_batch, img_prime_batch, feat_prime_batch = [], [], [], []
            for j in range(self.minibatch_size):
                transition = self.buffer.pop()
                s, a, r, s_prime, prob_a, done = transition
                img_batch.append(s[0])
                feat_batch.append(s[1])
                a_batch.append(a)
                r_batch.append([r])
                img_prime_batch.append(s_prime[0])
                feat_prime_batch.append(s_prime[1])
                prob_a_batch.append(prob_a)
                done_mask = 0 if done else 1
                done_batch.append(done_mask)
            
            img_batch = torch.squeeze(torch.tensor(img_batch), 1)
            feat_batch = torch.squeeze(torch.tensor(feat_batch), 1)
            img_prime_batch = torch.squeeze(torch.tensor(img_prime_batch), 1)
            feat_prime_batch = torch.squeeze(torch.tensor(feat_prime_batch), 1)
            mini_batch = [img_batch, feat_batch],  torch.tensor(a_batch, dtype=torch.float), torch.tensor(r_batch, dtype=torch.float), \
                [img_batch, feat_batch], torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)

            self.batch_data.append(mini_batch)
    
    
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
    
    
    def get_batch(self):
        """ object: 내부에서 _make_batch()를 호출하여 만든 batch data를 반환합니다.
        input: None
        output: self.batch_data -> List[[[List[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, 
                                        List[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], ... (#32)], ... (#10)]
        """
        self._make_batch()
        
        return self.batch_data
        
        
    def is_full(self):
        """ object: 현재 buffer가 minibatch를 만들 수 있을 정도로 가득 찬 상태인지 확인하고 그 결과를 반환합니다.
        input: None
        output: bool
        """
        flag = False
        if len(self.buffer) >= self.buffer_size * self.minibatch_size :
            flag = True
        
        return flag


class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.buffer = RolloutBuffer(config["buffer_size"], config["minibatch_size"])
        # parameter setting
        ### parameter for PPO
        self.mode = config["mode"]
        self.max_ep_len = config["max_ep_len"]
        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]
        self.lmbda = config["lmbda"]
        self.eps_clip = config["eps_clip"]     
        self.K_epochs = config["K_epochs"]               
        self.minibatch_size = config["minibatch_size"]   # M
        ### parameter for RAID
        self.action_num = 3
        self.layer_idx = config["layer_idx"]
        self.alpha = config["alpha"]
        self.model_name = config["model_name"]
        # PPO model setting   
        self.backbone_part1, self.backbone_part2 = self._split_model()
        self.shared_layer = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, 128),
            nn.ReLU()
        )
        self.channel = nn.Linear(128, 3) # discrete action : select channel
        self.index = nn.Linear(128, 2)   # continuous action1 : select index num
        self.noise = nn.Linear(128, 2)   # continuous action2 : select noise std
        self.critic = nn.Linear(128, 1)  # value network
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimization_step = 0
    
    
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
        agent_feature = self.backbone_part2(mid_feature + feature)
        agent_feature = torch.squeeze(agent_feature)
        
        return agent_feature
    
        
    def _policy(self, agent_feature, softmax_dim=0, batch=False):
        """ object: backbone을 통과하여 얻어진 feature를 각각의 policy layer를 통과시켜 action의 distribution을 계산해 반환합니다."""
        x = self.shared_layer(agent_feature)
        
        x1 = self.channel(x)
        x2 = self.index(x)
        x3 = self.noise(x)
        if batch :
            x2 = x2.transpose(0, 1)
            x3 = x3.transpose(0, 1)
        
        channel_prob = F.softmax(x1, dim=softmax_dim)
        channel_dist = Categorical(channel_prob)
        
        idx_mu = torch.sigmoid(x2[0])
        idx_std = F.softplus(x2[1])
        idx_dist = Normal(idx_mu, idx_std)
        
        noise_mu = torch.sigmoid(x3[0])
        noise_std = F.softplus(x3[1])
        noise_dist = Normal(noise_mu, noise_std)
        
        return channel_dist, idx_dist, noise_dist
    
    
    def _calc_advantage(self, data):
        """ object: 하나의 batch 안에 들어있는 각각의 mini_batch별로 Advantage를 계산하고 advantage를 추가한 batch를 만들어 반환합니다."""
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_probs = mini_batch

            td_target = r + self.gamma * self.get_value(s_prime) * done_mask
            delta = td_target - self.get_value(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_probs, td_target, advantage))

        return data_with_adv
    
    
    def get_value(self, state):
        """ object: input을 받아, 내부에서 _backbone 함수를 호출한 뒤 value를 반환합니다.
        *get_value는 Advantage를 계산할 때, 항상 batch단위로 호출되기 때문에 output이 Tensor형태입니다.
        input: state -> Tuple[torch.Tensor, torch.Tensor]
        output: value -> torch.Tensor[float]
        """
        img, feature = state
        with torch.no_grad():
            agent_feature = self._backbone(img, feature)
            x = self.shared_layer(agent_feature)
            v = self.critic(x)
        
        return v
    
    
    def get_actions(self, state, train=False):
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
                channel_dist, idx_dist, noise_dist = self._policy(agent_feature, softmax_dim=0)
        else :
            self.train()
            agent_feature = self._backbone(img, feature)
            channel_dist, idx_dist, noise_dist = self._policy(agent_feature, softmax_dim=1, batch=True)
        
        ch_a = channel_dist.sample()
        ch_log_prob = channel_dist.log_prob(ch_a)
        
        idx_a = idx_dist.sample()
        idx_log_prob = idx_dist.log_prob(idx_a)
        idx_a = torch.clamp(idx_a*1023, 0, 1023).int()
        
        std_a = noise_dist.sample()
        noise_log_prob = noise_dist.log_prob(std_a)
        std_a = torch.clamp(std_a*0.25, 0, 0.25)
        
        return (ch_a, idx_a, std_a), (ch_log_prob, idx_log_prob, noise_log_prob)


    def put_data(self, transition):
        """ object: 1번의 transition 데이터를 Tensor타입을 제거한 뒤 put()을 호출하여 buffer에 집어넣습니다.
        input: transition -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int, float], float, 
                                Tuple[torch.Tensor, torch.Tensor], Tuple[float, float, float], bool]
        output: None
        """
        s, a, r, s_prime, prob_a, done = transition
        s_noTensor = [data.tolist() for data in s]
        s_prime_noTensor = [data.tolist() for data in s_prime]
        self.buffer.put((s_noTensor, a, r, s_prime_noTensor, prob_a, done))

        
    def train_net(self):
        """ object: buffer가 가득차면 buffer에 쌓여있는 데이터를 사용하여 K_epochs번 DNN의 업데이트를 진행합니다.
        input: None
        output: None
        """
        if self.buffer.is_full() :
            data = self.buffer.get_batch()
            data = self._calc_advantage(data)

            for _ in range(self.K_epochs): 
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_probs, td_target, advantages = mini_batch
                    old_log_probs = old_log_probs.transpose(0, 1)
                    actions, log_probs = self.get_actions(s, train=True)
                    loss_list = []
                    
                    for i in range(self.action_num) :
                        ratio = torch.exp(log_probs[i] - old_log_probs[i])
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
                        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.get_value(s) , td_target)
                        loss_list.append(loss)

                    #TODO loss
                    loss = sum(loss_list)
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
                    