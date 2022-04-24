from torchvision import models
from kornia.losses import ssim_loss
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import piq
from functools import partial
from math import exp


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return {'tot': loss, 'l2': loss}


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # print(img1.shape, img2.shape)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class L2_SSIM_Loss(nn.Module):
    def __init__(self):
        super(L2_SSIM_Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        # self.ssim = ssim_loss

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        # ssim = 1 - self.ssim(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
            ssim = ssim_loss(inputs['rgb_fine'], targets, window_size=11)
            # ssim = 1 - self.ssim(inputs['rgb_fine'], targets)

        # ratio from MonoDepth
        return {'tot': loss + ssim * 2.8333, 'l2': loss, 'ssim': ssim}


class VGG16LossDirect(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(
            *list(vgg16.children())[0][:23])  # .to(self.device)
        for params in self.vgg.parameters():
            params.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, out, data):
        # print(out.shape, data.shape)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=out.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=out.device).reshape(1, 3, 1, 1)
        out = F.interpolate(out, size=(224, 224))
        out = (out - mean) / std
        data = F.interpolate(data, size=(224, 224))
        data = (data - mean) / std
        return self.l1(self.vgg(out), self.vgg(data))


class L2_vgg_Loss(nn.Module):
    def __init__(self):
        super(L2_vgg_Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.vgg = VGG16LossDirect()

    def forward(self, inputs, targets):
        # print(inputs['rgb_coarse'].shape, targets.shape)
        loss = self.loss(inputs['rgb_coarse'], targets)
        # vgg = self.vgg(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
            vgg = self.vgg(inputs['rgb_fine'], targets)

        return {'tot': loss + vgg * 0.01, 'l2': loss, 'vgg': vgg}


loss_dict = {'mse': MSELoss,
             'l2_ssim': L2_SSIM_Loss, 'l2_vgg': L2_vgg_Loss}
