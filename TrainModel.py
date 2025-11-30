import os
import scipy.stats
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from ResNet18_AE import ResNet18Enc, Quality
from ImageDataset import ImageDataset
from Transformers import AdaptiveResize


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor()
        ])
        # initialize the model
        self.test_batch_size = 1
        self.oracle_num = config.oracle_num
        self.sensitivity = Variable(torch.rand(1, self.oracle_num).cuda(), requires_grad=True)
        self.specificity = Variable(torch.rand(1, self.oracle_num).cuda(), requires_grad=True)
        print('number of annotators:{}'.format(self.config.oracle_num))
        self.netF = nn.DataParallel(ResNet18Enc(config)).cuda()
        self.netQ = nn.DataParallel(Quality()).cuda()

        # try load the model
        ckpt = self.get_latest_checkpoint(path=config.ckpt_path)
        self._load_checkpoint(ckpt=ckpt)

    def loader(self, csv_file, img_dir, transform, test, batch_size, shuffle, num_workers):
        data = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, test=test, cfg=self.config)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
        return loader

    def eval(self):
        val_loader = self.loader(csv_file=os.path.join(self.config.test_txt),
                                 img_dir=self.config.test_set, transform=self.test_transform, test=True,
                                 batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc, _, q_hat, q_mos = self.eval_once(loader=val_loader)

        print(srcc)
        return 0

    def eval_once(self, loader):
        q_mos, q_hat = [], []
        self.netF.eval()
        self.netQ.eval()
        for step, sample_batched in enumerate(loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x).cuda()
            y_bar = self.netQ(self.netF(x.squeeze(dim=0)))
            y_bar = y_bar.mean()
            y_bar.cpu()
            q_mos.append(y.data.numpy().item())
            q_hat.append(y_bar.cpu().data.numpy().item())
            if step % 100 == 0:
                print("completed:{}/{}".format(step, len(loader)))
        srcc = round(scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0], 3)
        plcc = round(scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0], 3)
        print(srcc, plcc)
        self.writemyfile(name='hat', list=q_hat)
        self.writemyfile(name='mos', list=q_mos)
        return srcc, plcc, q_hat, q_mos

    def writemyfile(self, name, list):
        Name = []
        Name.append(name)
        test = pd.DataFrame(columns=Name, data=list, index=None)
        test.to_csv(self.config.result_path + '/{}.csv'.format(name))

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.sensitivity = checkpoint['sensitivity']
            self.specificity = checkpoint['specificity']
            self.netF.load_state_dict(checkpoint['netF_dict'])
            self.netQ.load_state_dict(checkpoint['netQ_dict'])
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    def get_latest_checkpoint(self, path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])
