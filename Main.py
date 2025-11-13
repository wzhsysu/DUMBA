import argparse
import os, torch, random 
import numpy as np
import TrainModel
from utils import initialcode
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=True)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=19981025)
    parser.add_argument("--files", type=str, default="train_file", metavar='PATH', help='path to train file')
    parser.add_argument("--train_set", type=str, default="train_set", metavar='PATH', help='path to train dataset')
    parser.add_argument('--train_txt', type=str, default='train.txt')
    parser.add_argument("--val_set", type=str, default="val_set", metavar='PATH', help='path to validation dataset')
    parser.add_argument('--validation_txt', type=str, default='val.txt')
    parser.add_argument("--spaq_set", type=str, default="spaq_set", metavar='PATH', help='path to spaq dataset')
    parser.add_argument("--live_set", type=str, default="live_set", metavar='PATH', help='path to live dataset')
    parser.add_argument("--flive_set", type=str, default="flive_set", metavar='PATH', help='path to flive dataset')
    parser.add_argument("--csiq_set", type=str, default="csiq_set", metavar='PATH', help='path to csiq dataset')
    parser.add_argument("--livec_set", type=str, default="livec_set", metavar='PATH', help='path to livec dataset')
    parser.add_argument("--kadid_set", type=str, default="kadid_set", metavar='PATH', help='path to kadid dataset')
    parser.add_argument("--pipal_set", type=str, default="pipal_set", metavar='PATH', help='path to pipal dataset')
    parser.add_argument("--koniq_set", type=str, default="koniq_set", metavar='PATH', help='path to koniq dataset')
    parser.add_argument('--checkpoint', default="checkpoints", type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoint path')
    parser.add_argument("--oracle_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--fz", type=bool, default=True)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)
    parser.add_argument("--loss", type=str, default='DUMBA')
    parser.add_argument("--scheduler_milestones", type=int, nargs='+', default=[80])
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--projection", type=int, default=128)
    return parser.parse_args()

def main(cfg):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        print('train mode:{}'.format(cfg.train_txt))
        print(cfg)
        print('start training')
        t.fit()
    else:
        print('test mode')
        print('start testing')
        t.evaleveryepoch(epoch=1)
        return 0

if __name__ == "__main__":
    config = parse_config()
    config.train = True# modify when test
    print(initialcode(config=config))
    if config.train:
        config.scheduler_milestones = [1]
        config.fz = True
        config.batch_size = 64
        config.resume = False
        config.max_epochs = 1
        main(config)
        config.scheduler_milestones=[1,3,5,7,9]
        config.fz = False
        config.batch_size = 64
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = 10
        main(config)
    else:
        main(config)


