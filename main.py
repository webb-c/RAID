import argparse

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
    
    parser.add_argument("-lr", "--learningRate", type=float, default=0.0003, help="Learning Rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="Gamma")
    parser.add_argument("-clip", "--epsClip", type=float, default=0.2, help="epsilon value for Clip")
    parser.add_argument("-ep", "--epoch", type=int, default=10, help="Number of Epoch")
    parser.add_argument("-len", "--rolloutLength", type=int, default=3, help="Rollout Length")
    parser.add_argument("-buff", "--bufferSize", type=int, default=10, help="Buffer Size")
    parser.add_argument("-batch", "--minibatchSize", type=int, default=32, help="Minibatch Size")
 
    # parser.add_argument("-save", "--isSave", type=str2bool, default=False, help="Will you save your data?")
    # parser.add_argument("-test", "--idTest", type=str2bool, default=False, help="Are you going to test it?")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(conf):
    print(conf)
    print()
    """
    """
    
    
if __name__ == "__main__":
    args = parge_opt()
    conf = dict(**args.__dict__)
    main(conf)