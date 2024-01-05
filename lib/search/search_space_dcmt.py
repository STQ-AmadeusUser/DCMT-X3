import torch.nn as nn

from .operations import OPS
from .search_space_base import Conv1_1_Block, Block
from .search_space_base import Network as BaseSearchSpace
from lib.models.backbone.backbone import Backbones
from models.neck.neck import ShrinkChannel
from models.match.match import CFMN
from models.head.head import Cls_Reg


class Network(BaseSearchSpace):
    def __init__(self, config):
        super(Network, self).__init__(config)

        self.backbone = Backbones[config.MODEL.BACKBONE.NAME](affine=True)
        self.neck = ShrinkChannel(config.MODEL.NECK.IN_CHANNEL, config.MODEL.NECK.OUT_CHANNEL,
                                  backbone_name=config.MODEL.BACKBONE.NAME, affine=True)
        if config.MODEL.PRE_FUSION.NAME == 'CFMN':
            self.match = CFMN(config.MODEL.PRE_FUSION.IN_CHANNEL, config.MODEL.PRE_FUSION.HID_CHANNEL,
                              config.MODEL.PRE_FUSION.OUT_CHANNEL, config.MODEL.PRE_FUSION.STRIDE, affine=True)
        else:
            raise NotImplementedError

        self.fuse = nn.ModuleList()
        
        for i in range(self.num_blocks):
            input_config = self.input_configs[i]
            self.fuse.append(Block(
                input_config['in_chs'],
                input_config['ch'],
                input_config['strides'],
                input_config['num_stack_layers'],
                self.config
            ))

        self.collector = Conv1_1_Block(self.input_configs[-1]['in_chs'], config.SEARCH.LAST_DIM)
        self.prediction = Cls_Reg(config.SEARCH.LAST_DIM, affine=True)
        self.init_model(model_init=config.SEARCH.INIT.MODE)
        self.set_bn_param(config.SEARCH.INIT.BN_MOMENTUM, config.SEARCH.INIT.BN_EPS)
