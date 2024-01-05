import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from lib.models.activation import MishAuto as Mish


class FBNet(nn.Module):
    '''
    From LightTrack
    '''
    def __init__(self, model_name="fbnet_c", affine=True, track_running_stats=True):
        super(FBNet, self).__init__()
        model = fbnet(model_name, pretrained=False)

        # self.stages = nn.ModuleList([
        #     model.backbone.stages[:2],
        #     model.backbone.stages[2:5],
        #     model.backbone.stages[5:9],
        #     model.backbone.stages[9:18],
        # ])

        self.stages = model.backbone.stages[:18]

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
