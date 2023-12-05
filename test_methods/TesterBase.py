from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class TesterBase(ABC):

    def __init__(
            self,
            agent: nn.Module,
            env,
            conf,
            manager,
            agent_path = None) -> None:
        self.agent = agent
        
        if agent_path == None:
            self.random_policy = True
        else:
            self.agent.load_state_dict(torch.load(agent_path))
            self.random_policy = False

        self.env = env
        self.conf = conf
        self.manager = manager

    @abstractmethod
    def test(self):
        pass
