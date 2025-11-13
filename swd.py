from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def discrepancy_slice_wasserstein(p1, p2, num):
    s = p1.shape
    if s[1]>1:
        proj = torch.randn(s[1], num).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True)).cuda()
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1-p2
    wdist = torch.mean(torch.mul(dist, dist))
    return wdist