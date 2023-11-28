import warnings
import argparse
from tqdm import tqdm
from pyprnt import prnt
from agent.PPO import PPO
from agent.DQN import DQN
from Environment import Env
from Manager import Manager
from datetime import datetime

from defense.LocalGaussianBlurringDefense import LocalGaussianBlurringDefense, LocalGaussianBlurringDefensePolicy
from defense.MultiAugmentationDefense import MultiAugmentation, MultiAugmentationPolicy

from train_methods.PPOTrainer import PPOTrainer
from train_methods.DQNTrainer import DQNTrainer

import copy

def str2bool(v):
    """ object: command line 인자 중 bool 타입을 판별하기 위한 함수입니다."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 't'):
        return True
    elif v.lower() in ('false', 'no', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_opt(known=False):
    """ object: command line 인자를 전달받기 위한 함수입니다."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-mode", "--mode", type=str, default="train", help="train / val / test")
    parser.add_argument("-eps", "--max_ep_len", type=int, default=10, help="max timesteps in one episode")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="learning rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help=" discount factor gamma")
    parser.add_argument("-lmbda", "--lmbda", type=float, default=0.9, help="hyperparameter lambda for cal GAE")
    parser.add_argument("-clip", "--eps_clip", type=float, default=0.2, help="clip parameter for PPO")
    
    parser.add_argument("-episode", "--num_episode", type=int, default=30000, help="number of train episode")
    parser.add_argument("-epoch", "--num_epoch", type=int, default=10, help="number of maximum action")
    parser.add_argument("-step", "--num_step", type=int, default=10, help="number of PPO's step" )
    parser.add_argument("-Kepoch", "--K_epochs", type=int, default=3, help="update policy for K Epoch")
    parser.add_argument("-buff", "--buffer_size", type=int, default=10, help="buffer size")
    parser.add_argument("-batch", "--minibatch_size", type=int, default=32, help="minibatch size")
    
    parser.add_argument("-layer", "--layer_idx", type=int, default=4, help="feature extract layer in mobilenet")
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="hyperparameter alpha for cal Reward")
    parser.add_argument("-name", "--model_name", type=str, default="mobilenet", help="attacked DNN model name")
    parser.add_argument("-dataset", "--dataset_name", type=str, default="CIFAR10", help="train dataset name")
    parser.add_argument("-attack", "--train_attack", type=str, default="PGDLinf", help="attack type to make attacked images")

    parser.add_argument("-save", "--image_save", action='store_true', default=False, help="save step images")
    parser.add_argument("-imgheight", "--image_height", type=int, default=32, help="image height")
    parser.add_argument("-imgwidth", "--image_width", type=int, default=32, help="image width")

    
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


#TODO val / test 모드 전환
def main(conf):
    """ object: PPO 알고리즘을 사용하여 Attacked Image를 defense하는 policy를 Agent에게 학습시킵니다."""
    manager = Manager(use=True)
    prnt(conf)
    
    # Hyper-parameter
    mode = conf["mode"]

    
    # Env, Agent setting
    env = Env(conf, MultiAugmentation)
    agent = PPO(conf, MultiAugmentationPolicy)
    trainer = PPOTrainer(agent, env, conf, manager)
    
    # Train code
    if mode == "train" : 
        
        trainer.train()
            
        print("Train finish with,")
        prnt(conf)
        print("Start:\t", manager.get_time().strftime('%m-%d %H:%M:%S'))
        print("End:\t", datetime.now().strftime('%m-%d %H:%M:%S'))


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    args = parse_opt()
    conf = dict(**args.__dict__)
    
    main(conf)