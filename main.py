import warnings
import argparse
from tqdm import tqdm
from pyprnt import prnt
from Agent import Agent
from Environment import Env

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 't'):
        return True
    elif v.lower() in ('false', 'no', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-mode", "--mode", type=str, default="train", help="train / val / test")
    parser.add_argument("-eps", "--max_ep_len", type=int, default=10, help="max timesteps in one episode")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="learning rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help=" discount factor gamma")
    parser.add_argument("-lmbda", "--lmbda", type=float, default=0.9, help="hyperparameter lambda for cal GAE")
    parser.add_argument("-clip", "--eps_clip", type=float, default=0.2, help="clip parameter for PPO")
    
    parser.add_argument("-episode", "--num_episode", type=int, default=1000, help="number of train episode")
    parser.add_argument("-epoch", "--num_epoch", type=int, default=10, help="number of maximum action")
    parser.add_argument("-step", "--num_step", type=int, default=10, help="number of PPO's step" )
    parser.add_argument("-Kepoch", "--K_epochs", type=int, default=3, help="update policy for K Epoch")
    parser.add_argument("-buff", "--buffer_size", type=int, default=10, help="buffer size")
    parser.add_argument("-batch", "--minibatch_size", type=int, default=32, help="minibatch size")
    
    parser.add_argument("-layer", "--layer_idx", type=int, default=4, help="feature extract layer in mobilenet")
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="hyperparameter alpha for cal Reward")
    parser.add_argument("-name", "--model_name", type=str, default="mobilenet", help="attacked DNN model name")
    parser.add_argument("-dataset", "--dataset_name", type=str, default="CIFAR10", help="train dataset name")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

#TODO Train / Test 모드 전환
def main(conf):
    print("===== Experiment Setting =====")
    prnt(conf)
    print()
    
    # Hyper-parameter
    num_episode = conf["num_episode"]
    num_step = conf["num_step"] 
    interval = 20
    
    # Env, Agent setting
    env = Env(conf)
    env.train()
    agent = Agent(conf)
    
    # Train code
    for episode in tqdm(range(num_episode)):
        total_reward = 0
        state, _ = env.reset()
        done = False
        for step in range(num_step):
            actions, action_probs = agent.get_actions(state)
            state_prime, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated :
                done = True
            agent.put_data((state, actions, reward, state_prime, action_probs, done))
            total_reward += reward
            state = state_prime
            if done : 
                break
        agent.train_net()
        
        if episode % interval == 0 :
            print("# of episode :{}, avg reward : {:.2f}, total reward : {:.2f}".format(episode, total_reward/step, total_reward))
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    args = parse_opt()
    conf = dict(**args.__dict__)
    main(conf)