import argparse
from Agent import Agent

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 't'):
        return True
    elif v.lower() in ('false', 'no', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parge_opt(known=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="Learning Rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="Gamma")
    parser.add_argument("-lmbda", "--lmbda", type=float, default=0.9, help="Lambda") #?
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="alpha for Reward")
    parser.add_argument("-clip", "--eps_clip", type=float, default=0.2, help="epsilon value for Clip")
    
    parser.add_argument("-epoch", "--num_epoch", type=int, default=10, help="Number of Epoch")
    parser.add_argument("-step", "--num_step", type=int, default=50, help="Number of Step" )
    parser.add_argument("-len", "--rollout_len", type=int, default=3, help="Rollout Length")
    parser.add_argument("-buff", "--buffer_size", type=int, default=10, help="Buffer Size")
    parser.add_argument("-batch", "--minibatch_size", type=int, default=32, help="Minibatch Size")
    
    parser.add_argument("-mode", "--mode", type=str, default="train", help="train / val / test")
    parser.add_argument("-name", "--model_name", type=str, default="mobilenet", help="DNN model name")
    parser.add_argument("-dataset", "--dataset_name", type=str, default="CIFAR10", help="train dataset name")
    parser.add_argument("-layer", "--layer_idx", type=int, default=4, help="feature extract layer in mobilenet")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(conf):
    print(conf)
    print()
    
    
if __name__ == "__main__":
    args = parge_opt()
    conf = dict(**args.__dict__)
    main(conf)