import _init_paths
import os
import torch
import argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import utils.read_file as reader
from dataset.siamese_builder_deploy import CalibrationDataset as data_builder


def parse_args():
    parser = argparse.ArgumentParser(description='Generate DCMT Calibration Data')
    parser.add_argument('--arch_type', type=str, default='lt', help='DCMT from which type')
    args = parser.parse_args()
    args.cfg = '../experiments/DCMT_{}/Retrain_{}.yaml'.format(args.arch_type, args.arch_type)
    args.calib_z = "./calibration_{}/template/".format(args.arch_type)
    args.calib_x = "./calibration_{}/search/".format(args.arch_type)
    args.calib_b = "./calibration_{}/template_bbox/".format(args.arch_type)
    args.calib_path = [args.calib_z, args.calib_x, args.calib_b]
    return args


def calibration():
    # preprocess and configure
    args = parse_args()
    config = edict(reader.load_yaml(args.cfg))
    for calib_path in args.calib_path:
        if not os.path.exists(calib_path): os.makedirs(calib_path)

    # build dataset
    train_set = data_builder(config)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8,
                              pin_memory=True, sampler=None, drop_last=True)

    for iter_id, batch_data in enumerate(train_loader):
        template = batch_data['template']  # bx3x128x128
        search = batch_data['search']  # bx3x256x256
        bbox = batch_data['template_bbox']  # bx4

        print('template shape: ', template.shape)
        print('search shape: ', search.shape)
        print('template bbox shape: ', bbox.shape)

        if iter_id < 128:
            # template calibration data
            z = template.numpy().astype(np.uint8)
            z.tofile(args.calib_path[0] + "z" + "_" + str(iter_id) + ".bin")
            # search calibration data
            x = search.numpy().astype(np.uint8)
            x.tofile(args.calib_path[1] + "x" + "_" + str(iter_id) + ".bin")
            # template bbox calibration data
            roi = bbox.numpy().astype(np.uint8)
            roi = np.expand_dims(roi, -1)
            roi = np.expand_dims(roi, -1)
            roi.tofile(args.calib_path[2] + "b" + "_" + str(iter_id) + ".bin")
        else:
            break


if __name__ == '__main__':
    calibration()
