import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.distributions import Normal, Categorical
from torchvision.models import mobilenet_v2

from main import parse_opt

class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.data = []
        # parameter setting
        ### parameter for PPO
        self.mode = config["mode"]
        self.max_ep_len = config["max_ep_len"]
        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]
        self.lmbda = config["lmbda"]
        self.eps_clip = config["eps_clip"]
        # self.rollout_len = config["rollout_len"]         
        self.K_epochs = config["K_epochs"]               # K
        self.buffer_size = config["buffer_size"]         # NT
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

    
    def _load_model(self, path=None): 
        # from utils.py -> 추후 제거
        if self.model_name == 'mobilenet':
            model = mobilenet_v2(weights='IMAGENET1K_V1')
            num_ftrs = model.classifier._modules["1"].in_features
            model.classifier._modules["1"] = torch.nn.Linear(num_ftrs, 10)
            model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            if path is not None:
                model.load_state_dict(torch.load(path))
        
        return model
    
    def _split_model(self):
        backbone = self._load_model()
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

    
    def _backbone(self, image, feature):
        mid_feature = self.backbone_part1(image)
        # print(feature.shape, mid_feature.shape)
        agent_feature = self.backbone_part2(mid_feature + feature)
        return agent_feature.flatten()
    
        
    def _policy(self, agent_feature, softmax_dim=0):
        x = self.shared_layer(agent_feature)
        
        x1 = self.channel(x)
        channel_prob = F.softmax(x1, dim=softmax_dim)
        channel_dist = Categorical(channel_prob)
        
        x2 = self.index(x)
        idx_mu = torch.tanh(x2[0])
        idx_std = F.softplus(x2[1])
        idx_dist = Normal(idx_mu, idx_std)
        
        x3 = self.noise(x)
        noise_mu = torch.tanh(x3[0])
        noise_std = F.softplus(x1[1])
        noise_dist = Normal(noise_mu, noise_std)
        
        return channel_dist, idx_dist, noise_dist
    
    
    def _value(self, agent_feature):
        x = self.shared_layer(agent_feature)
        v = self.critic(x)
        
        return v
    
    
    def get_actions(self, state, train=False, softmax_dim=0):
        """ object : input을 받아, 내부에서 _backbone, _policy 함수를 호출한 뒤 action과 해당 action의 log_prob들을 tuple로 반환한다.
        input : state -> Tuple, softmax_dim -> int
        output : actions -> Tuple, probs -> Tuple
        """
        if not train:
            self.backbone_part1.eval()
            self.backbone_part2.eval()
            with torch.no_grad():
                agent_feature = self._backbone(state[0], state[1])
                channel_dist, idx_dist, noise_dist = self._policy(agent_feature, softmax_dim)
        else :
            self.backbone_part1.train()
            self.backbone_part2.train()
            agent_feature = self._backbone(state[0], state[1])
            channel_dist, idx_dist, noise_dist = self._policy(agent_feature, softmax_dim)
        
        ch_a = channel_dist.sample()
        ch_log_prob = channel_dist.log_prob(ch_a)
        idx_a = idx_dist.sample()
        idx_log_prob = idx_dist.log_prob(idx_a)
        std_a = noise_dist.sample()
        noise_log_prob = noise_dist.log_prob(std_a)
        return (ch_a, idx_a, std_a), (ch_log_prob, idx_log_prob, noise_log_prob)


    def put_data(self, transition):
        s, a, r, s_prime, prob_a, done = transition
        s_noTensor = [data.tolist() for data in s]
        s_prime_noTensor = [data.tolist() for data in s_prime]
        self.data.append((s_noTensor, a, r, s_prime_noTensor, prob_a, done))
    
    
    def _make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        batch_data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in self.data:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append(prob_a)
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                
            print(type(s[0]))
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)

            
            batch_data.append(mini_batch)

        return batch_data

    def _calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_probs = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self._value(s_prime) * done_mask
                delta = td_target - self._value(s)
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

        
    def train_net(self):
        """ object : buffer가 가득차면 buffer에 쌓여있는 데이터를 사용하여 K_epochs번 DNN의 업데이트를 진행합니다.
        input : None
        output : None
        """
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self._make_batch()
            data = self._calc_advantage(data)

            for _ in range(self.K_epochs):  # 이 횟수만큼 저장된 데이터를 사용하여 학습한다.
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_probs, td_target, advantages = mini_batch

                    actions, log_probs = self.get_actions(s, True, softmax_dim=1)
                    loss_list = []
                    for i in range(self.action_num) :
                        ratio = torch.exp(log_probs[i] - old_log_probs[i])
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
                        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)
                        loss_list.append(loss)

                    #TODO loss
                    loss = sum(loss_list)
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
                    
# for testing
if __name__ == '__main__':
    args = parse_opt()
    conf = dict(**args.__dict__)
    model = Agent(conf)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        state = (torch.rand(1, 3, 32, 32), torch.rand(1, 32, 16, 16))
        state_prme = (torch.rand(1, 3, 32, 32), torch.rand(1, 32, 16, 16))
        r = 10
        done = False
        while not done:
            for t in range(20):
                actions, action_probs = model.get_actions(state)
                r = 10
                model.put_data((state, actions, r, state_prme, action_probs, done))

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
