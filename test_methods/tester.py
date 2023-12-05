import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from test_methods.TesterBase import TesterBase

class Tester(TesterBase):
    def __init__(self, agent: nn.Module, env, conf, manager, agent_path = None) -> None:
        super(Tester, self).__init__(agent, env, conf, manager, agent_path)

    def test(self):
        # Hyper-parameter
        num_episode = self.conf["num_episode"]
        num_step = self.conf["num_step"] 
        mode = self.conf["mode"]
        rand = self.conf["rand"]
        print_interval = 100
        save_interval = 100

        self.env.test()
        
        # Train code
        score = 0
        dataset_num = 0
        
        for episode in tqdm(range(num_episode)):
            dataset_num += 1
            state, _ = self.env.reset()
            if state == -1: # 모든 이미지를 보았으면, break
                break

            initial_confidence, _ = self.env.inference()
            initial_label = np.argmax(initial_confidence)

            self.manager.save_image(episode, 0, state[0]) # 변화 없는 이미지 = 0
            if self.conf['action_logger']:
                self.manager.save_action(episode, 0, [], [], [], epi=True)
        
            done = False
            for step in range(num_step):
                actions, action_probs, entropies = self.agent.get_actions(state, rand=rand)
                
                state_prime, reward, terminated, truncated, info = self.env.step(actions)
                if terminated or truncated :
                    done = True
                state = state_prime

                self.manager.save_image(episode, step+1, state[0])

                if self.conf['action_logger']:
                    self.manager.save_action(episode, step+1, actions, action_probs, entropies, epi=False)

                if done: # truncated만 잡히는 조건
                    step_confidence, _ = self.env.inference()
                    step_label = np.argmax(step_confidence)
                    if initial_label == step_label:
                        result = initial_label
                    else:
                        result = step_label
                        
                    break
    
            if result == self.env.target_label:
                score += 1
        
        print("\n# of data :{}, correction ratio : {:.2f}".format(dataset_num, score/dataset_num))
        self.manager.record(mode+"/accuracy", score/dataset_num, 1)
