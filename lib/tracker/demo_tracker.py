import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import utils.tracking_helper as tracking_helper


class DemoTracker(object):
    def __init__(self, config):
        super(DemoTracker, self).__init__()
        self.config = config

    def init(self, inputs):
        """
        initilaize the Siamese tracking networks
        """

        # parse inputs
        im, self.target_pos, self.target_sz, self.model = inputs['image'], inputs['pos'], inputs['sz'], inputs['model']

        p = DefaultConfig()
        self.im_h = im.shape[0]
        self.im_w = im.shape[1]
        p.update({'MODEL_NAME': self.config.MODEL.NAME})
        p.update({'BACKBONE_NAME': self.config.MODEL.BACKBONE.NAME})
        p.renew()
        
        # hyperparameters
        cfg_benchmark = self.config.DEMO.HYPERS
        p.update(cfg_benchmark)
        p.renew()

        if 'small_sz' in cfg_benchmark.keys():
            if ((self.target_sz[0] * self.target_sz[1]) / float(self.im_h * self.im_w)) < 0.004:
                p.instance_size = cfg_benchmark['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_benchmark['small_sz']
                p.renew()

        self.p = p

        self.window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        self.grids(p)

        # crop image for Siamese
        wc_z = self.target_sz[0] + self.p.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.p.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        self.avg_chans = np.mean(im, axis=(0, 1))

        crop_input = {'image': im, 'pos': self.target_pos, 'model_sz': self.p.exemplar_size, 'original_sz': s_z, 'avg_chans': self.avg_chans}
        z_crop_meta = tracking_helper.siam_crop(crop_input)
        z_crop, z_crop_info = z_crop_meta['image_tensor'], z_crop_meta['meta_info']

        mask = tracking_helper.generate_psedou_mask(self.target_pos, self.target_sz, (self.im_h, self.im_w))

        crop_input['image'] = mask
        mask_crop_meta = tracking_helper.siam_crop(crop_input, mode='numpy')
        mask_crop = (mask_crop_meta['image_tensor'] > 0.5).astype(np.uint8)
        mask_crop = torch.from_numpy(mask_crop)
        target_box = tracking_helper.get_bbox(s_z, self.p, self.target_sz)

        self.model.template({'template': z_crop.unsqueeze(0).cuda(), 'template_mask': mask_crop.unsqueeze(0).cuda(),
                             'target_box': target_box})

    def track(self, im):
        # crop image in subsequent frames
        hc_z = self.target_sz[1] + self.p.context_amount * sum(self.target_sz)
        wc_z = self.target_sz[0] + self.p.context_amount * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        self.scale_z = self.p.exemplar_size / s_z
        d_search = (self.p.instance_size - self.p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / self.scale_z
        s_x = s_z + 2 * pad
        crop_input = {'image': im, 'pos': self.target_pos, 'model_sz': self.p.instance_size,
                      'original_sz': tracking_helper.python2round(s_x),
                      'avg_chans': self.avg_chans}
        x_crop_meta = tracking_helper.siam_crop(crop_input)

        target_sz_incrop = self.target_sz * self.scale_z
        x_crop, x_crop_info = x_crop_meta['image_tensor'], x_crop_meta['meta_info']

        # tracking and update state
        x_crop = x_crop.unsqueeze(0).cuda()

        outputs = self.model.track({'search': x_crop})

        cls_score, bbox_pred = outputs['cls'], outputs['reg']
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz_incrop)))  # scale penalty
        r_c = self.change((target_sz_incrop[0] / target_sz_incrop[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * self.p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - self.p.window_influence) + self.window * self.p.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - self.p.instance_size // 2
        diff_ys = pred_ys - self.p.instance_size // 2

        diff_xs, diff_ys = diff_xs / self.scale_z, diff_ys / self.scale_z
        pred_w, pred_h = pred_w / self.scale_z, pred_h / self.scale_z

        target_sz_inimg = target_sz_incrop / self.scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * self.p.lr

        # size rate
        res_xs = self.target_pos[0] + diff_xs
        res_ys = self.target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz_inimg[0]
        res_h = pred_h * lr + (1 - lr) * target_sz_inimg[1]

        self.target_pos = np.array([res_xs, res_ys])
        self.target_sz = np.array([res_w, res_h])

        self.target_pos[0] = max(0, min(self.im_w, self.target_pos[0]))
        self.target_pos[1] = max(0, min(self.im_h, self.target_pos[1]))
        self.target_sz[0] = max(10, min(self.im_w, self.target_sz[0]))
        self.target_sz[1] = max(10, min(self.im_h, self.target_sz[1]))

        return {'pos': self.target_pos, 'sz': self.target_sz, 'score': pscore[r_max, c_max]}

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def IOUgroup(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy

        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class DefaultConfig(object):
    MODEL_NAME = 'DCMT'
    BACKBONE_NAME = 'LightTrack'
    ABLATION = '-1'
    penalty_k = 0.023
    window_influence = 0.16
    lr = 0.8
    windowing = 'cosine'
    context_amount = 0.5
    ratio = 0.94

    if BACKBONE_NAME == 'ResNet50Dilated':
        exemplar_size = 127
        instance_size = 255
        total_stride = 8
        score_size = (instance_size - exemplar_size) // total_stride + 15
    elif BACKBONE_NAME in ['FBNet', 'LightTrack']:
        exemplar_size = 128
        instance_size = 256
        total_stride = 16
        if ABLATION != '1':
            score_size = (instance_size - exemplar_size) // total_stride + 8
        else:  # Ablation1 has different score size
            instance_size = 320
            score_size = (instance_size - exemplar_size) // total_stride + 1
    elif BACKBONE_NAME == 'WaveMLP':
        exemplar_size = 128
        instance_size = 256
        total_stride = 16
        score_size = (instance_size - exemplar_size) // total_stride + 8
    elif BACKBONE_NAME == 'AlexNet':
        exemplar_size = 127
        instance_size = 287
        total_stride = 8
        score_size = (instance_size - exemplar_size) // total_stride + 6
    else:
        raise Exception('Unknown backbone!')

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        if self.BACKBONE_NAME == 'ResNet50Dilated':
            self.total_stride = 8
            if self.ABLATION != '1':
                if self.instance_size in range(255, 263):
                    self.score_size = 31
                elif self.instance_size in range(263, 271):
                    self.score_size = 32
                elif self.instance_size in range(271, 279):
                    self.score_size = 33
                elif self.instance_size in range(279, 287):
                    self.score_size = 34
                elif self.instance_size in range(287, 295):
                    self.score_size = 35
                else:
                    raise NotImplementedError
            else:
                if self.instance_size in range(255, 263):
                    self.score_size = 25
                elif self.instance_size in range(263, 271):
                    self.score_size = 26
                elif self.instance_size in range(271, 279):
                    self.score_size = 27
                elif self.instance_size in range(279, 287):
                    self.score_size = 28
                elif self.instance_size in range(287, 295):
                    self.score_size = 29
                else:
                    raise NotImplementedError
        elif self.BACKBONE_NAME == 'LightTrack':
            self.total_stride = 16
            if self.ABLATION != '1':
                if self.instance_size == 256:
                    self.score_size = 16
                elif self.instance_size in range(257, 273):
                    self.score_size = 17
                elif self.instance_size in range(273, 289):
                    self.score_size = 18
                elif self.instance_size in range(289, 305):
                    self.score_size = 19
                elif self.instance_size in range(305, 321):
                    self.score_size = 20
                elif self.instance_size in range(321, 337):
                    self.score_size = 21
                else:
                    raise NotImplementedError
            else:  # Ablation1 has different score size
                if self.instance_size == 256:
                    self.score_size = 9
                elif self.instance_size in range(257, 273):
                    self.score_size = 10
                elif self.instance_size in range(273, 289):
                    self.score_size = 11
                elif self.instance_size in range(289, 305):
                    self.score_size = 12
                elif self.instance_size in range(305, 321):
                    self.score_size = 13
                elif self.instance_size in range(321, 337):
                    self.score_size = 14
                else:
                    raise NotImplementedError
        elif self.BACKBONE_NAME == 'WaveMLP':
            self.total_stride = 16
            if self.instance_size in range(256, 259):
                self.score_size = 16
            elif self.instance_size in range(259, 275):
                self.score_size = 17
            elif self.instance_size in range(275, 291):
                self.score_size = 18
            elif self.instance_size in range(291, 307):
                self.score_size = 19
            elif self.instance_size in range(307, 323):
                self.score_size = 20
            else:
                raise NotImplementedError
        elif self.BACKBONE_NAME == 'AlexNet':
            self.total_stride = 8
            if self.instance_size in range(255, 263):
                self.score_size = 22
            elif self.instance_size in range(263, 271):
                self.score_size = 23
            elif self.instance_size in range(271, 279):
                self.score_size = 24
            elif self.instance_size in range(279, 287):
                self.score_size = 25
            elif self.instance_size in range(287, 295):
                self.score_size = 26
            elif self.instance_size in range(295, 303):
                self.score_size = 27
            elif self.instance_size in range(303, 311):
                self.score_size = 28
            elif self.instance_size in range(311, 319):
                self.score_size = 29
            else:
                raise NotImplementedError
        else:
            raise Exception('Unknown backbone!')
