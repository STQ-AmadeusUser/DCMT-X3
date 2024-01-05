import torch
import torch.nn as nn
from lib.models.activation import MishAuto as Mish


class Cls_Reg(nn.Module):
    """
    original cross-correlation head used in SiamFC, SiamDW
    """
    def __init__(self, dim=256, towernum=4, affine=True):
        super(Cls_Reg, self).__init__()

        bbox_tower = []  # for regression
        cls_tower = []   # for classification
        # regression head
        for i in range(towernum):
            if i == 0:
                bbox_tower.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1))
            else:
                bbox_tower.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1))

            bbox_tower.append(nn.GroupNorm(32, dim, affine=affine))
            bbox_tower.append(nn.ReLU())

        # classification head
        for i in range(towernum):
            if i == 0:
                cls_tower.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1))
            else:
                cls_tower.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1))

            cls_tower.append(nn.GroupNorm(32, dim, affine=affine))
            cls_tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        # prediction layers
        self.bbox_pred = nn.Conv2d(dim, 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cls_pred = nn.Conv2d(dim, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # adjust scale
        adjust = nn.Parameter(0.1 * torch.ones(1))
        bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))
        self.register_parameter('adjust', adjust)
        self.register_parameter('bias', bias)

    def forward(self, response):

        # predictions
        x_reg = self.bbox_tower(response)

        r = getattr(self, 'adjust') * self.bbox_pred(x_reg) + getattr(self, 'bias')
        reg = torch.exp(r)

        x_cls = self.cls_tower(response)
        cls = 0.1 * self.cls_pred(x_cls)

        return cls, reg
