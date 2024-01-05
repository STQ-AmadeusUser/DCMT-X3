import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.models.activation import MishAuto as Mish


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs=64, hid_chs=16):
        super(SqueezeExcite, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, hid_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(hid_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * x_se.sigmoid()
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs=16, dw_chs=16, dw_k=3, dw_s=1, dw_p=1, se_chs=8, out_chs=16):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = (dw_s == 1 and in_chs == out_chs)
        self.conv_dw = nn.Conv2d(in_chs, dw_chs, dw_k, dw_s, dw_p, groups=dw_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(dw_chs)
        # Squeeze-and-excitation
        self.se = SqueezeExcite(dw_chs, se_chs)
        self.conv_pw = nn.Conv2d(dw_chs, out_chs, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = x.mul(x.sigmoid())
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_residual:
            x += residual
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs=16, dw_chs=64, dw_k=7, dw_s=2, dw_p=3, se_chs=16, out_chs=24):
        super(InvertedResidual, self).__init__()
        self.has_residual = (in_chs == out_chs and dw_s == 1)

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_chs, dw_chs, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dw_chs)

        # Depth-wise convolution
        self.conv_dw = nn.Conv2d(dw_chs, dw_chs, dw_k, dw_s, dw_p,
                                 groups=dw_chs, bias=False)
        self.bn2 = nn.BatchNorm2d(dw_chs)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(dw_chs, se_chs)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv2d(dw_chs, out_chs, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = x.mul(x.sigmoid())

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = x.mul(x.sigmoid())

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            x += residual

        return x


class LightTrack(nn.Module):

    def __init__(self, affine=True, track_running_stats=True):
        super(LightTrack, self).__init__()

        # Stem
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Middle stages (IR/ER/DS Blocks)
        blocks = []
        # stage 0
        blocks.append(nn.Sequential(
            DepthwiseSeparableConv(in_chs=16, dw_chs=16, dw_k=3, dw_s=1, dw_p=1, se_chs=8, out_chs=16)
        ))
        # stage 1
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=16, dw_chs=64, dw_k=7, dw_s=2, dw_p=3, se_chs=16, out_chs=24),
            InvertedResidual(in_chs=24, dw_chs=144, dw_k=7, dw_s=1, dw_p=3, se_chs=40, out_chs=24),
        ))
        # stage 2
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=24, dw_chs=96, dw_k=3, dw_s=2, dw_p=1, se_chs=24, out_chs=40),
            InvertedResidual(in_chs=40, dw_chs=160, dw_k=5, dw_s=1, dw_p=2, se_chs=40, out_chs=40),
            InvertedResidual(in_chs=40, dw_chs=240, dw_k=7, dw_s=1, dw_p=3, se_chs=64, out_chs=40),
            InvertedResidual(in_chs=40, dw_chs=240, dw_k=3, dw_s=1, dw_p=1, se_chs=64, out_chs=40),
        ))
        # stage 3
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=40, dw_chs=160, dw_k=7, dw_s=2, dw_p=3, se_chs=40, out_chs=80),
            InvertedResidual(in_chs=80, dw_chs=320, dw_k=3, dw_s=1, dw_p=1, se_chs=80, out_chs=80),
            InvertedResidual(in_chs=80, dw_chs=320, dw_k=7, dw_s=1, dw_p=3, se_chs=80, out_chs=80),
            InvertedResidual(in_chs=80, dw_chs=320, dw_k=7, dw_s=1, dw_p=3, se_chs=80, out_chs=80),
        ))
        # stage 4
        blocks.append(nn.Sequential(
            InvertedResidual(in_chs=80, dw_chs=480, dw_k=7, dw_s=1, dw_p=3, se_chs=120, out_chs=96),
            InvertedResidual(in_chs=96, dw_chs=384, dw_k=5, dw_s=1, dw_p=2, se_chs=96, out_chs=96),
            InvertedResidual(in_chs=96, dw_chs=576, dw_k=3, dw_s=1, dw_p=1, se_chs=144, out_chs=96),
        ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # architecture = [[0], [], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = x.mul(x.sigmoid())
        x = self.blocks(x)
        return x
