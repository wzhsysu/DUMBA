import argparse
import os, torch, random
import numpy as np
import TrainModel
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=199810)
    parser.add_argument('--test_txt', type=str, default='test.txt') # need to revise for testing
    parser.add_argument("--test_set", type=str, default="path to test dataset") # need to revise for testing
    parser.add_argument('--checkpoint', default="checkpoints", type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--result_path', default="results", type=str, help='checkpoint path')
    parser.add_argument("--oracle_num", type=int, default=10)
    parser.add_argument("--fz", type=bool, default=True)
    return parser.parse_args()

def main(cfg):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    t = TrainModel.Trainer(cfg)
    print('test mode')
    print('start testing')
    t.eval()
    return 0

if __name__ == "__main__":
    config = parse_config()
    config.ckpt_path = config.checkpoint
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    main(config)
