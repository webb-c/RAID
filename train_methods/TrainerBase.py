from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class TrainerBase(ABC):

    def __init__(self, agent: nn.Module, env, conf, manager, agent_path = None) -> None:
        self.agent = agent
        
        if agent_path != None:
            self.agent.load_state_dict(torch.load(agent_path))

        self.env = env
        self.conf = conf
        self.manager = manager


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def test(self):
        pass
