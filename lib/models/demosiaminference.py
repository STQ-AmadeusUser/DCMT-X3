import matplotlib.pyplot as plt
import torch
import importlib
import torch.nn as nn


class DemoSiamInference(nn.Module):
    def __init__(self, derived_model=None, config=None):
        super(DemoSiamInference, self).__init__()
        self.cfg = config
        self.init_arch(derived_model)

    def init_arch(self, derived_model):
        self.backbone = derived_model.backbone
        self.neck = derived_model.neck
        self.match = derived_model.match
        self.fuse = derived_model.fuse
        self.collector = derived_model.collector
        self.prediction = derived_model.prediction

    # only for testing
    def template(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - template_mask: BHW (optional)
        """

        template = inputs['template']
        zf = self.backbone(template)
        zf_neck = self.neck(zf)
        self.zf = zf_neck

        if 'target_box' in inputs.keys():
            self.target_box = torch.tensor(inputs['target_box'], dtype=torch.float32).to(self.zf.device)
            self.target_box = self.target_box.view(1, 4)

    def track(self, inputs):
        """
        inputs:
         - search: BCHW, H*W:255*255
        """

        search = inputs['search']
        xf = self.backbone(search)
        xf_neck = self.neck(xf)

        # match
        merge = self.match(self.zf, xf_neck, self.target_box)

        # fuse
        for i, block in enumerate(self.fuse):
            merge = block(merge)

        # collector
        map = self.collector(merge)

        # prediction
        cls_pred, reg_pred = self.prediction(map)

        preds = {'reg': reg_pred, 'cls': cls_pred}

        return preds
