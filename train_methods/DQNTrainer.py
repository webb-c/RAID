from train_methods.TrainerBase import TrainerBase
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

class DQNTrainer(TrainerBase):

    def __init__(self, agent: nn.Module, env, conf, manager, agent_path = None) -> None:
        super(DQNTrainer, self).__init__(agent, env, conf, manager, agent_path)

    def train(self):
        # Hyper-parameter
        num_episode = self.conf["num_episode"]
        num_step = self.conf["num_step"] 
        mode = self.conf["mode"]
        print_interval = 100
        save_interval = 100
        q_target_interval = 1000

        self.env.train()

        q_target = copy.deepcopy(self.agent)
        
        # Train code

        total_reward = 0
        for episode in tqdm(range(num_episode)):
            if episode % q_target_interval == 0:
                q_target = copy.deepcopy(self.agent)
            epi_reward = 0
            state, _ = self.env.reset()
            if self.conf['image_save'] and episode%save_interval==0:
                self.manager.save_image(episode, 0, state[0]) # 변화 없는 이미지 = 0
            done = False
            for step in range(num_step):
                actions = self.agent.get_actions(state)
                state_prime, reward, terminated, truncated, info = self.env.step(actions)
                if terminated or truncated :
                    done = True
                self.agent.put_data((state, actions, reward, state_prime, done))
                reward = reward.item()
                epi_reward += reward
                state = state_prime
                if self.conf['image_save'] and episode%save_interval==0:
                    self.manager.save_image(episode, step+1, state[0])
                if done : 
                    break
            total_reward += epi_reward
            # record total_reward & avg_reward & loss for each episode
            self.manager.record(mode+"/total_reward", epi_reward, episode)
            self.manager.record(mode+"/avg_reward", (epi_reward/(step+1)), episode)

            loss = self.agent.train_net(q_target)
        
            if loss is not None :
                self.manager.record(mode+"/loss", loss.mean(), episode)
                
            if episode % print_interval == 0 and step != 0:
                print("\n# of episode :{}, avg reward : {:.2f}, total reward : {:.2f}".format(episode, total_reward/print_interval, total_reward))
                total_reward = 0

    def eval(self):
        pass

    def test(self):
        pass