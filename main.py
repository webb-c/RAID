import warnings
import argparse
from tqdm import tqdm
from pyprnt import prnt
from Agent import Agent
from Environment import Env
from Manager import Manager
from datetime import datetime

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

    parser.add_argument("-save", "--image_save", action='store_true', default=False, help="save step images")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


#TODO val / test 모드 전환
def main(conf):
    """ object: PPO 알고리즘을 사용하여 Attacked Image를 defense하는 policy를 Agent에게 학습시킵니다."""
    manager = Manager(use=True)
    prnt(conf)
    
    # Hyper-parameter
    num_episode = conf["num_episode"]
    num_step = conf["num_step"] 
    mode = conf["mode"]
    print_interval = 100
    save_interval = 500
    
    # Env, Agent setting
    env = Env(conf)
    env.train()
    env.set_log_path(manager.get_log_path()) # 동적인 인자이므로 init할 때 주지 않고 지금 줌
    agent = Agent(conf)
    
    # Train code
    if mode == "train" : 
        total_reward = 0
        for episode in tqdm(range(num_episode)):
            epi_reward = 0
            state, _ = env.reset()
            if conf['image_save'] and episode%save_interval==0:
                manager.save_image(episode, 0, state[0]) # 변화 없는 이미지 = 0
            done = False
            for step in range(num_step):
                actions, action_probs = agent.get_actions(state)
                state_prime, reward, terminated, truncated, info = env.step(actions)
                if terminated or truncated :
                    done = True
                agent.put_data((state, actions, reward, state_prime, action_probs, done))
                reward = reward.item()
                epi_reward += reward
                state = state_prime
                if conf['image_save'] and episode%save_interval==0:
                    manager.save_image(episode, step+1, state[0])
                if done : 
                    break
            loss = agent.train_net()
            total_reward += epi_reward
            # record total_reward & avg_reward & loss for each episode
            manager.record(mode+"/total_reward", epi_reward, episode)
            manager.record(mode+"/avg_reward", (epi_reward/(step+1)), episode)
            if loss is not None :
                manager.record(mode+"/loss", loss.mean().item(), episode)
            if episode % print_interval == 0 and step != 0:
                print("\n# of episode :{}, avg reward : {:.2f}, total reward : {:.2f}".format(episode, 100*(epi_reward/print_interval), 100*epi_reward))
                epi_reward = 0
    
    print("Train finish with,")
    prnt(conf)
    print("Start:\t", manager.get_time().strftime('%m-%d %H:%M:%S'))
    print("End:\t", datetime.now().strftime('%m-%d %H:%M:%S'))


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    args = parse_opt()
    conf = dict(**args.__dict__)
    
    main(conf)