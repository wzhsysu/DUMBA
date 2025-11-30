from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch.autograd import Function
import numpy as np
# from utils.qpsolver import landmarks

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ResizeConv2d':
        init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, val=0)

class ResNet18Enc(nn.Module):
    def __init__(self, config):
        super(ResNet18Enc, self).__init__()
        self.ResNet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if config.fz == True:
            # Freeze all previous layers.
            for param in self.ResNet18.parameters():
                param.requires_grad = False
        self.cfg=config
    def forward(self, x):
        x = self.ResNet18.conv1(x)
        x = self.ResNet18.bn1(x)
        x = self.ResNet18.relu(x)
        x = self.ResNet18.maxpool(x)
        x = self.ResNet18.layer1(x)
        x = self.ResNet18.layer2(x)
        x = self.ResNet18.layer3(x)
        x = self.ResNet18.layer4(x) #512 7 7
        x = self.ResNet18.avgpool(x).view(x.size(0), -1)
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm

class Quality(nn.Module):
    def __init__(self):
        super(Quality, self).__init__()
        self.linear = nn.Linear(512, 64, bias=False)
        self.qualityfc = nn.Linear(64, 1, bias=False)
        self.linear.apply(weights_init)
        self.qualityfc.apply(weights_init)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        quality = 3*torch.tanh(self.qualityfc(x))
        return quality

class Landmark(nn.Module):
    def __init__(self):
        super(Landmark, self).__init__()
        layers = [nn.Linear(512, 256, bias=False),
                  nn.ReLU(),
                  nn.Linear(256, 1, bias=False)
                ]
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)
    def forward(self,x):
        x=self.net(x)
        return torch.relu(x)/torch.sum(torch.relu(x),dim=0)


class Domain(nn.Module):
    def __init__(self):
        super(Domain, self).__init__()
        #for DANN
        layers = [nn.Linear(512, 256, bias=True),
                  nn.ReLU(),
                  nn.Linear(256, 128, bias=True),
                  nn.ReLU(),
                  nn.Linear(128, 1, bias=True),
                  nn.Sigmoid()
                ]
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        domain = self.net(x)
        return domain.view(-1,1).squeeze(1)
        
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
if __name__ == "__main__":
    for i in range(5):
        fs = torch.rand(256, 512).cuda()
        ft = torch.rand(256, 512).cuda()
        NetL=Landmark().cuda()
        x,y=NetL(fs), NetL(ft)
        # print(x)
        # y=x*fs
        # print(y)
        print(x.norm(),y.norm())