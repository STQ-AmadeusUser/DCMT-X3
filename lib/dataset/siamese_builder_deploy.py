from __future__ import division

import os
import cv2
import json
import math
import random
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import utils.box_helper as boxhelper
import utils.augmentation as auger

import sys
sys.path.append('../')


sample_random = random.Random()


class CalibrationDataset(Dataset):
    def __init__(self, cfg):
        super(CalibrationDataset, self).__init__()
        # pair information
        self.template_size = cfg.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.TRAIN.SEARCH_SIZE
        self.stride = cfg.MODEL.PRE_FUSION.STRIDE
        self.score_size = cfg.TRAIN.SCORE_SIZE
        self.cfg = cfg

        # data augmentation
        self.color = cfg.TRAIN.DATASET.AUG.COMMON.COLOR
        self.flip = cfg.TRAIN.DATASET.AUG.COMMON.FLIP
        self.rotation = cfg.TRAIN.DATASET.AUG.COMMON.ROTATION
        self.blur = cfg.TRAIN.DATASET.AUG.COMMON.BLUR
        self.gray = cfg.TRAIN.DATASET.AUG.COMMON.GRAY
        self.label_smooth = cfg.TRAIN.DATASET.AUG.COMMON.LABELSMOOTH
        self.mixup = cfg.TRAIN.DATASET.AUG.COMMON.MIXUP
        self.neg = 0.0
        self.jitter = None

        self.shift_s = cfg.TRAIN.DATASET.AUG.SEARCH.SHIFTs
        self.scale_s = cfg.TRAIN.DATASET.AUG.SEARCH.SCALEs
        self.shift_e = cfg.TRAIN.DATASET.AUG.EXEMPLAR.SHIFT
        self.scale_e = cfg.TRAIN.DATASET.AUG.EXEMPLAR.SCALE

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
            + ([transforms.Grayscale(num_output_channels=3), ] if self.gray > random.random() else [])
        )

        # train data information
        print('train datas: {}'.format(cfg.TRAIN.DATASET.WHICH_USE))
        self.train_datas = []  # all train dataset
        start = 0
        self.num = 0
        for data_name in cfg.TRAIN.DATASET.WHICH_USE:
            dataset = subData(cfg, data_name, start)
            self.train_datas.append(dataset)
            start += dataset.num  # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        videos_per_epoch = cfg.TRAIN.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        template, search = dataset._get_pairs(index, dataset.data_name)

        template, search = self.check_exists(index, dataset, template, search)
        template, search = self.check_damaged(index, dataset, template, search)

        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, bbox_t, dag_param_t = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size, search=True)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        outputs = {'template': template,
                   'search': search,
                   'template_bbox': np.array(bbox_t, np.float32),
                   }

        outputs = self.data_package(outputs)

        return outputs

    def data_package(self, outputs):
        clean = []
        for k, v in outputs.items():
            if v is None:
                clean.append(k)
        if len(clean) == 0:
            return outputs
        else:
            for k in clean:
                del outputs[k]

        return outputs

    def check_exists(self, index, dataset, template, search):
        name = dataset.data_name
        while True:
            if not (os.path.exists(template[0]) and os.path.exists(search[0])):
                index = random.randint(0, 100)
                template, search = dataset._get_pairs(index, name)
                continue
            else:
                return template, search

    def check_damaged(self, index, dataset, template, search):
        name = dataset.data_name
        while True:
            if cv2.imread(template[0]) is None or cv2.imread(search[0]) is None:
                index = random.randint(0, 100)
                template, search = dataset._get_pairs(index, name)
                continue
            else:
                return template, search

    def _shuffle(self):
        """
        random shuffel
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = boxhelper.center2corner(boxhelper.Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _mixupRandom(self):
        """
        gaussian random -- 0.3~0.7
        """
        return random.random() * 0.4 + 0.3

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size, search=False):
        """
        data augmentation for input pairs
        """
        shape = image.shape
        crop_bbox = boxhelper.center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if search:
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_s), (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            param.shift = (self._posNegRandom() * self.shift_e, self._posNegRandom() * self.shift_e)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_e), (1.0 + self._posNegRandom() * self.scale_e))  # scale change

        crop_bbox, _ = auger.aug_apply(boxhelper.Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = boxhelper.BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = boxhelper.Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)  # other data augmentation
        return image, bbox, param

    def _mixupShift(self, image, size):
        """
        random shift mixed-up image
        """
        shape = image.shape
        crop_bbox = boxhelper.center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * 64, self._posNegRandom() * 64)  # shift
        crop_bbox, _ = boxhelper.aug_apply(boxhelper.Corner(*crop_bbox), param, shape)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image


# ---------------------
# for a single dataset
# ---------------------
class subData(object):
    """
    for training with multi dataset
    """

    def __init__(self, cfg, data_name, start):
        self.data_name = data_name
        self.start = start

        info = cfg.TRAIN.DATASET.CONFIG[data_name]
        self.frame_range = info.RANGE
        self.num_use = info.USE
        self.root = info.PATH

        with open(info.ANNOTATION) as fin:
            self.labels = json.load(fin)
            self._clean()
            self.num = len(self.labels)  # video numer

        self._shuffle()

    def _clean(self):
        """
        remove empty videos/frames/annos in dataset
        """
        # no frames
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))

        for video, track in to_del:
            try:
                del self.labels[video][track]
            except:
                pass

        # no track/annos
        to_del = []

        if self.data_name == 'YTB':
            to_del.append('train/1/YyE0clBPamU')  # This video has no bounding box.
        print(self.data_name)

        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_image_anno(self, video, track, frame):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)

        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def _get_pairs(self, index, data_name):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            track_info.pop('frames')
        except KeyError:
            pass
        frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames) - 1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]

        if data_name in ['VISDRONE_DET', 'VISDRONE_VID', 'PURDUE']:
            search_frame = template_frame
            template_frame = int(frames[template_frame])
            search_frame = int(frames[search_frame])
        else:
            template_frame = int(frames[template_frame])
            search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _get_negative_target(self, index=-1):
        """
        dasiam neg
        """
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        try:
            track_info.pop('frames')
        except KeyError:
            pass
        frames = list(track_info.keys())
        frame = int(random.choice(frames))

        return self._get_image_anno(video_name, track, frame)

    def _get_hard_negative_target(self, index, data_name):  # for VISDRONE_DET, VISDRONE_VID
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_list = list(video.keys())

        if data_name == 'VISDRONE_DET':
            track = random.choice(track_list)
            track_info = video[track]
            try:
                track_info.pop('frames')
            except KeyError:
                pass
            frames = list(track_info.keys())

            template_frame = random.randint(0, len(frames) - 1)

            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames) - 1) + 1
            search_range = frames[left:right]

            template_frame = int(frames[template_frame])

            if template_frame == int(search_range[0]) and template_frame == int(search_range[-1]):
                return None, None

            while True:
                search_frame = int(random.choice(search_range))
                if template_frame != search_frame:
                    break

            return self._get_image_anno(video_name, track, template_frame), \
                   self._get_image_anno(video_name, track, search_frame)

        elif data_name in ['VISDRONE_VID', 'PURDUE']:
            if len(track_list) <= 1:
                return None, None
            template_track_id = random.randint(0, len(track_list) - 1)
            template_track = track_list[template_track_id]

            left = max(template_track_id - self.frame_range, 0)
            right = min(template_track_id + self.frame_range, len(track_list) - 1) + 1
            search_range = track_list[left:right]
            while True:
                search_track = random.choice(search_range)
                if template_track != search_track:
                    break
            # template sample
            template_track_info = video[template_track]
            try:
                template_track_info.pop('frames')
            except KeyError:
                pass
            template_frame = random.choice(list(template_track_info.keys()))
            # search sample
            search_track_info = video[search_track]
            try:
                search_track_info.pop('frames')
            except KeyError:
                pass
            search_frame = random.choice(list(search_track_info.keys()))

            return self._get_image_anno(video_name, template_track, int(template_frame)), \
                   self._get_image_anno(video_name, search_track, int(search_frame))

        else:
            raise NotImplementedError
