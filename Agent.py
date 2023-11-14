import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torchvision.models import mobilenet_v2

class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.data = []
        # parameter
        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]
        self.eps_clip = config["eps_clip"]
        self.layer_num = config["layer_num"]
        self.mode = config["mode"]
        self.rollout_len = config["rollout_len"]
        self.self.buffer_size = config["self.buffer_size"]
        self.minibatch_size = config["minibatch_size"]
        self.action_number = 2  # std & index
        
        # PPO DNN model
        self.backbone = self.load_model("mobilenet")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.policy = nn.Sequential(
            nn.Linear(1280, 500),
            nn.ReLU(),
            nn.Linear(500, self.action_number)
        )
        
        self.fc1   = nn.Linear(3,128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.fc_v = nn.Linear(128,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimization_step = 0

    def _load_model(model_name, path=None): 
        # in utils.py
        if model_name == 'mobilenet':
            model = mobilenet_v2(weights='IMAGENET1K_V1')
            num_ftrs = model.classifier._modules["1"].in_features
            model.classifier._modules["1"] = torch.nn.Linear(num_ftrs, 10)
            model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            if path is not None:
                model.load_state_dict(torch.load(path))
        return model
    
    def _backbone():
        """"""
        
    def _policy():
        """"""
        
    def get_action(self, image, feature):
        agent_feature = self._backbone(image, feature)
        prob = self._polciy(agent_feature)
        
        return np.argmax(prob)
        
    
    def _pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def _v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def _put_data(self, transition):
        self.data.append(transition)
        
    def _make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                          torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                          torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data

    def _calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1