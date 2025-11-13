import os, time
import scipy.stats
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from MNL_Loss import Binary_Loss, mmd_loss, CORAL
import prettytable as pt
from ResNet18_AE import ResNet18Enc, Quality, Domain, Landmark
from ImageDataset import ImageDataset
from Transformers import AdaptiveResize
from utils import get_latest_checkpoint, save_checkpoint
from swd import discrepancy_slice_wasserstein
from cdist import cdist

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(3),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                      std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                      std=(0.229, 0.224, 0.225))
        ])
        # initialize the model
        self.train_batch_size = config.batch_size
        self.test_batch_size = 1
        self.initial_lr = config.lr
        self.oracle_num = config.oracle_num
        self.sensitivity = Variable(torch.rand(1, self.oracle_num).cuda(), requires_grad=True)
        self.specificity = Variable(torch.rand(1, self.oracle_num).cuda(), requires_grad=True)
        print('number of annotators:{}'.format(self.config.oracle_num))
        self.netF = nn.DataParallel(ResNet18Enc(config)).cuda()
        self.netQ = nn.DataParallel(Quality()).cuda()
        if self.config.loss == 'dann':
            self.netD = nn.DataParallel(Domain()).cuda()
            self.optimizer = optim.Adam([
            {'params': self.netF.parameters(), 'lr': self.initial_lr},
            {'params': self.netQ.parameters(), 'lr': self.initial_lr},
            {'params': self.netD.parameters(), 'lr': self.initial_lr},
            {'params': self.sensitivity, 'lr': 1e-3},
            {'params': self.specificity, 'lr': 1e-3}])
        elif self.config.loss == 'DUMBA' or self.config.loss == 'DUMBA_mmd':
            self.netL = nn.DataParallel(Landmark()).cuda()
            self.optimizer = optim.Adam([
            {'params': self.netF.parameters(), 'lr': self.initial_lr},
            {'params': self.netQ.parameters(), 'lr': self.initial_lr},
            {'params': self.sensitivity, 'lr': 1e-3},
            {'params': self.specificity, 'lr': 1e-3}])
            self.optimizerL = optim.Adam([{'params': self.netL.parameters(), 'lr': self.initial_lr}])
        else:
            self.optimizer = optim.Adam([
            {'params': self.netF.parameters(), 'lr': self.initial_lr},
            {'params': self.netQ.parameters(), 'lr': self.initial_lr},
            {'params': self.sensitivity, 'lr': 1e-3},
            {'params': self.specificity, 'lr': 1e-3}])
           
        self.model_name = 'DA'
        self.bce_fn = nn.BCELoss().cuda()
        self.loss_fn = Binary_Loss().cuda()
        self.mmd_loss = mmd_loss().cuda()
        
        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.mse = torch.nn.MSELoss()
        self.train_loss = []
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.result_path = config.result_path
        self.preds_path = config.preds_path
        self.best_validation_srcc = 0.0

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                  last_epoch=self.start_epoch-1,
                                                  milestones=config.scheduler_milestones,
                                                  gamma=config.scheduler_gamma)
        if self.config.loss == 'DUMBA':
            self.schedulerL = lr_scheduler.MultiStepLR(self.optimizerL,
                                                       last_epoch=self.start_epoch-1,
                                                       milestones=config.scheduler_milestones,
                                                       gamma=config.scheduler_gamma)
        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt)
            else:
                ckpt = get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)
     
        print('scheduler:', config.scheduler_milestones, 'gamma:', config.scheduler_gamma)

    def loader(self, csv_file, img_dir, transform, test, batch_size, shuffle, num_workers):
        data = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, test=test, cfg=self.config)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,pin_memory=True, num_workers=num_workers)
        return loader

    def fit(self):
        # torch.autograd.set_detect_anomaly(True)
        self.best_srcc = 0
        self.best_epoch = 0
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_loader = self.loader(csv_file=os.path.join(self.config.files, self.config.train_txt),
                                            img_dir=self.config.train_set, transform=self.train_transform, test=False,
                                            batch_size=self.train_batch_size, shuffle=True, num_workers=16)
            _ = self._train_single_epoch(epoch)
        if epoch!=0:
            srcc_m = 0
            maxepoch = 0
            val_loader = self.loader(csv_file=os.path.join(self.config.validation_txt),
                                     img_dir=self.config.val_set, transform=self.test_transform, test=True,
                                     batch_size=self.test_batch_size, shuffle=False, num_workers=1)
            for i in range(1, self.max_epochs):
                self._load_checkpoint(self.config.checkpoint+'/checkpoints/DA-'+'%05d.pt'%(i))
                sr, pl, _, _ = self.eval_once(loader=val_loader)
                print('val: srcc: {:.4f} plcc: {:.4f}'.format(sr, pl))
                if sr > srcc_m:
                    srcc_m = sr
                    maxepoch = i
                print('max srcc:{:.4f} epoch:{}'.format(srcc_m, maxepoch))

            print('epoch{} is best epoch!!'.format(maxepoch))
            self._load_checkpoint(self.config.checkpoint+'/checkpoints/DA-'+'%05d.pt' % (maxepoch))
            _, _ = self.eval(maxepoch)

    def evaleveryepoch(self, epoch):

        val_loader = self.loader(csv_file=os.path.join(self.config.validation_txt),
                                 img_dir=self.config.val_set, transform=self.test_transform, test=True,
                                 batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc, _, q_hat, q_mos = self.eval_once(loader=val_loader)

        print(srcc)
        return 0
        
    def _train_single_epoch(self, epoch):
        time_s = time.time()
        loader_steps = len(self.train_loader)
        start_steps = epoch * len(self.train_loader)
        total_steps = self.config.max_epochs * len(self.train_loader)
        local_counter = epoch * total_steps + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1][0]
        running_loss_q = 0 if epoch == 0 else self.train_loss[-1][1]
        running_loss_s = 0 if epoch == 0 else self.train_loss[-1][3]
        running_loss_k = 0 if epoch == 0 else self.train_loss[-1][4]
        loss_corrected, loss_q_corrected, loss_r_corrected, loss_s_corrected, loss_k_corrected = 0.0, 0.0, 0.0, 0.0, 0.0
        running_duration = 0.0
        # start training
        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            s1, s2, g, t1, t2 = sample_batched['s1'], sample_batched['s2'],sample_batched['y'], \
                sample_batched['t1'], sample_batched['t2']

            s1, s2, g, t1, t2 = Variable(s1).cuda(), Variable(s2).cuda(), \
                 Variable(g).cuda(), Variable(t1).cuda(), Variable(t2).cuda()

            x = torch.cat([s1, s2, t1, t2], dim=0)
            if self.config.loss == 'DUMBA':
                self.optimizerL.zero_grad()
            self.optimizer.zero_grad()
            f = self.netF(x) #feature generator
            y = self.netQ(f) #quality prediction
            fs, ft = f[:(s1.shape[0]+s2.shape[0])], f[(s1.shape[0]+s2.shape[0]):]
            ys, yt = y[:(s1.shape[0]+s2.shape[0])], y[(s1.shape[0]+s2.shape[0]):]
            y1, y2 = ys[:s1.shape[0]], ys[s1.shape[0]:]
            y_diff = y1 - y2
            y_var = torch.ones_like(y1) + torch.ones_like(y2) + 1e-4
            p = 0.5 * (1 + torch.erf(y_diff/torch.sqrt(2*y_var)))
            self.loss_q = self.loss_fn(p, g, torch.sigmoid(self.sensitivity), torch.sigmoid(self.specificity))
            
            if self.config.loss == 'DUMBA':
                # a = self.netL(fs.detach()).squeeze(1)
                a = ot.unif(fs.shape[0])
                b = ot.unif(ft.shape[0])
                g_dist = cdist(fs, ft, 'ecli')**2
                q_dist = cdist(ys.detach(), yt, 'fid')
                M = g_dist + q_dist
                # M = g_dist
                # pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a,b,M.detach().cpu().numpy(),0.001,reg_m=10)
                pi = ot.unbalanced.mm_unbalanced(a, b, M.detach().cpu().numpy(), reg_m=10)
                pi = torch.from_numpy(pi).float().cuda()
                print(torch.sum(pi))
                self.loss_s = torch.sum(pi*M)
                if step<10:
                    print(pi)
                    print(self.loss_s)
                self.loss_k = self.loss_s - self.loss_s

            if self.config.loss == 'ot':
                a, b = torch.tensor(ot.unif(fs.shape[0])).cuda(), torch.tensor(ot.unif(ft.shape[0])).cuda()
                g_dist = cdist(fs, ft, 'ecli')**2
                q_dist = cdist(ys.detach(), yt, 'fid')
                dist = 0.2*g_dist + q_dist
                self.loss_s = ot.emd2(a, b, dist)
                self.loss_k = self.loss_s - self.loss_s
            if self.config.loss == 'swd':
                self.loss_s = 0.2*discrepancy_slice_wasserstein(fs, ft, self.config.projection)
                self.loss_k = 0.02*discrepancy_slice_wasserstein(ys.detach(), yt, self.config.projection)
            if self.config.loss == 'base':
                self.loss_s = self.loss_q - self.loss_q
                self.loss_k = self.loss_s - self.loss_s
               
            self.loss = self.loss_q + self.loss_s + self.loss_k
            self.loss.backward()
            self.optimizer.step()
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)
            running_loss_q = beta * running_loss_q + (1 - beta) * self.loss_q.data.item()
            loss_q_corrected = running_loss_q / (1 - beta ** local_counter)
            running_loss_s = beta * running_loss_s + (1 - beta) * self.loss_s.data.item()
            loss_s_corrected = running_loss_s / (1 - beta ** local_counter)
            running_loss_k = beta * running_loss_k + (1 - beta) * self.loss_k.data.item()
            loss_k_corrected = running_loss_k / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d/%d) [Loss = %.4f, Loss_q = %.5f, Loss_cf = %.5f, Loss_cpl= %.4f] (%.1f samples/sec; %.3f sec/batch)')
            print(format_str % (epoch, step, loader_steps, loss_corrected, loss_q_corrected, loss_s_corrected, loss_k_corrected*10000, \
                                examples_per_sec, duration_corrected))    
            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append([loss_corrected, loss_q_corrected, loss_r_corrected, loss_s_corrected, loss_k_corrected])
        print('finish train: {}'.format(time.time()-time_s))
        self.netF.eval()
        self.netQ.eval()
        config = self.config
        val_loader = self.loader(csv_file=os.path.join(config.validation_txt),
                                 img_dir=config.val_set, transform=self.test_transform, test=True,
                                 batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        sr, pl, _, _ = self.eval_once(loader=val_loader)
        print('val: srcc: {:.4f} plcc: {:.4f}'.format(sr, pl))
        if sr > self.best_srcc:
            self.best_srcc = sr
            self.best_epoch = epoch
        print('best srcc:{} best epoch:{}'.format(self.best_srcc, self.best_epoch))
        del val_loader

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            if self.config.loss == 'dann':
                save_checkpoint({
                    'epoch': epoch,
                    'netF_dict': self.netF.state_dict(),
                    'netQ_dict': self.netQ.state_dict(),
                    'netD_dict': self.netD.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'sensitivity': self.sensitivity,
                    'specificity': self.specificity,
                    'train_loss': self.train_loss
                }, model_name)
            elif self.config.loss == 'DUMBA':
                save_checkpoint({
                    'epoch': epoch,
                    'netF_dict': self.netF.state_dict(),
                    'netQ_dict': self.netQ.state_dict(),
                    'netL_dict': self.netL.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'optimizerL': self.optimizerL.state_dict(),
                    'sensitivity': self.sensitivity,
                    'specificity': self.specificity,
                    'train_loss': self.train_loss
                }, model_name)
            else:
                save_checkpoint({
                    'epoch': epoch,
                    'netF_dict': self.netF.state_dict(),
                    'netQ_dict': self.netQ.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'sensitivity': self.sensitivity,
                    'specificity': self.specificity,
                    'train_loss': self.train_loss
                }, model_name)
            print('save:' + model_name)
        else:
            print('not save epoch:{}'.format(epoch))
        self.scheduler.step()
        if self.config.loss == 'DUMBA':
            self.schedulerL.step()
        return 1
        
    def eval_once(self, loader):
        q_mos, q_hat = [],[]
        self.netF.eval()
        self.netQ.eval()
        for step, sample_batched in enumerate(loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x).cuda()
            y_bar= self.netQ(self.netF(x.squeeze(dim=0)))
            y_bar = y_bar.mean()
            y_bar.cpu()
            q_mos.append(y.data.numpy().item())
            q_hat.append(y_bar.cpu().data.numpy().item())
            if step%100 == 0:
                print("completed:{}/{}".format(step, len(loader)))
        srcc = round(scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0], 3)
        plcc = round(scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0], 3)
        print(srcc, plcc)
        self.writemyfile(name='s_hat', list = q_hat)
        self.writemyfile(name='s_mos', list = q_mos)
        return srcc, plcc, q_hat, q_mos

    def writemyfile(self, name, list):
        Name=[]
        Name.append(name)
        test = pd.DataFrame(columns=Name, data=list, index=None)
        test.to_csv(self.config.result_path+'/{}.csv'.format(name))
  
    def eval(self, epoch):
        srcc, plcc = {}, {}
        config = self.config
         # testing set configuration
        print('Evaluating...epoch{}'.format(epoch))
        val_loader = self.loader(csv_file=os.path.join(config.validation_txt),
                                 img_dir=config.val_set, transform=self.test_transform, test=True,
                                 batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['validation'], plcc['validation'], q_hat, q_mos = self.eval_once(loader=val_loader)
        print('val: srcc: {:.4f} plcc: {:.4f}'.format(srcc['validation'], plcc['validation']))
        self.writemyfile(name='val_hat', list=q_hat)
        self.writemyfile(name='val_mos', list=q_mos)
        del val_loader

        spaq_loader = self.loader(csv_file=os.path.join(config.files, 'spaq_png.txt'),
                                  img_dir=config.spaq_set, transform=self.test_transform, test=True,
                                  batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['spaq'], plcc['spaq'], q_hat, q_mos = self.eval_once(loader=spaq_loader)
        print('spaq: srcc: {:.4f} plcc: {:.4f}'.format(srcc['spaq'], plcc['spaq']))
        self.writemyfile(name='spaq_hat', list=q_hat)
        self.writemyfile(name='spaq_mos', list=q_mos)
        del spaq_loader

        livec_loader = self.loader(csv_file=os.path.join(config.files, 'livec_test.txt'),
                                   img_dir=config.livec_set, transform=self.test_transform, test=True,
                                   batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['livec'], plcc['livec'], q_hat, q_mos = self.eval_once(loader=livec_loader)
        print('livec: srcc: {:.4f} plcc: {:.4f}'.format(srcc['livec'], plcc['livec']))
        self.writemyfile(name='livec_hat', list=q_hat)
        self.writemyfile(name='livec_mos', list=q_mos)
        del livec_loader

        koniq_loader = self.loader(csv_file=os.path.join(config.files, 'koniq.txt'),
                                   img_dir=config.koniq_set, transform=self.test_transform, test=True,
                                   batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['koniq'], plcc['koniq'], q_hat, q_mos = self.eval_once(loader=koniq_loader)
        print('koniq: srcc: {:.4f} plcc: {:.4f}'.format(srcc['koniq'], plcc['koniq']))
        self.writemyfile(name='koniq_hat', list=q_hat)
        self.writemyfile(name='koniq_mos', list=q_mos)
        del koniq_loader

        csiq_loader = self.loader(csv_file=os.path.join(config.files, 'csiq.txt'),
                                  img_dir=config.csiq_set, transform=self.test_transform, test=True,
                                  batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['csiq'], plcc['csiq'], q_hat, q_mos = self.eval_once(loader=csiq_loader)
        print('csiq: srcc: {:.4f} plcc: {:.4f}'.format(srcc['csiq'], plcc['csiq']))
        self.writemyfile(name='csiq_hat', list=q_hat)
        self.writemyfile(name='csiq_mos', list=q_mos)
        del csiq_loader

        pipal_loader = self.loader(csv_file=os.path.join(config.files, 'pipal.txt'),
                                   img_dir=config.pipal_set, transform=self.test_transform, test=True,
                                   batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['pipal'], plcc['pipal'], q_hat, q_mos = self.eval_once(loader=pipal_loader)
        print('pipal: srcc: {:.4f} plcc: {:.4f}'.format(srcc['pipal'], plcc['pipal']))
        self.writemyfile(name='pipal_hat', list=q_hat)
        self.writemyfile(name='pipal_mos', list=q_mos)
        del pipal_loader

        kadid_loader = self.loader(csv_file=os.path.join(config.files, 'kadid_test.txt'),
                                   img_dir=config.kadid_set, transform=self.test_transform, test=True,
                                   batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['kadid'], plcc['kadid'], q_hat, q_mos= self.eval_once(loader=kadid_loader)
        print('kadid: srcc: {:.4f} plcc: {:.4f}'.format(srcc['kadid'], plcc['kadid']))
        self.writemyfile(name='kadid_hat', list=q_hat)
        self.writemyfile(name='kadid_mos', list=q_mos)
        del kadid_loader

        live_loader = self.loader(csv_file=os.path.join(config.files, 'live_test.txt'),
                                  img_dir=config.live_set, transform=self.test_transform, test=True,
                                  batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        srcc['live'], plcc['live'], q_hat, q_mos = self.eval_once(loader=live_loader)
        print('live: srcc: {:.4f} plcc: {:.4f}'.format(srcc['live'], plcc['live']))
        self.writemyfile(name='live_hat', list=q_hat)
        self.writemyfile(name='live_mos', list=q_mos)
        del live_loader

        del config
        with open(os.path.join(self.result_path, 'result_{}.txt'.format(epoch)), 'w') as txt_file:
            tb = pt.PrettyTable()
            tb.field_names = ["---", "spaq", "pipal", "LIVE", "koniq", "CSIQ", "LIVEC", "KADID10K", "VALIDATION"]
            tb.add_row(['SRCC', srcc['spaq'], srcc['pipal'], srcc['live'], srcc['koniq'], srcc['csiq'], srcc['livec'], srcc['kadid'], srcc['validation']])
            tb.add_row(['PLCC', plcc['spaq'], plcc['pipal'], plcc['live'], plcc['koniq'], plcc['csiq'], plcc['livec'], plcc['kadid'], plcc['validation']])
            print(tb)
            txt_file.write(str(tb))
        return srcc, plcc

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.sensitivity = checkpoint['sensitivity']
            self.specificity = checkpoint['specificity']
            self.start_epoch = checkpoint['epoch']+1
            # self.start_epoch = 0
            self.train_loss = checkpoint['train_loss']
            self.netF.load_state_dict(checkpoint['netF_dict'])
            self.netQ.load_state_dict(checkpoint['netQ_dict'])
            # if self.config.loss == 'dann':
            #     self.netD.load_state_dict(checkpoint['netD_dict'])
            # if self.config.loss == 'DUMBA':
            #     self.netL.load_state_dict(checkpoint['netL_dict'])
            #     self.optimizerL.load_state_dict(checkpoint['optimizerL'])
            # self.model.load_state_dict(checkpoint['state_dict'],strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})  restart epoch{}!"
                  .format(ckpt, checkpoint['epoch'],self.start_epoch))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))
