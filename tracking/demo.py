import _init_paths
import pickle
import argparse
import cv2
import torch
import numpy as np
from easydict import EasyDict as edict

import tracker.demo_tracker as tracker_builder
import utils.box_helper as boxhelper
import utils.model_helper as loader
import utils.read_file as reader
import search.model_derived as model_derived
import utils.video_helper as player
from models.demosiaminference import DemoSiamInference

torch.set_num_threads(1)
# if 'DISPLAY' not in os.environ:
#     os.environ['DISPLAY'] = 'localhost:12.0'


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--arch_type', type=str, default='lt', help='demo DCMT from which type')
    parser.add_argument('--resume',  default='../snapshot/DCMT_LightTrack_GOT10KTEST.pth', help='resume checkpoint')
    parser.add_argument('--video', default='../video/ChasingDrones', type=str, help='videos or image files')
    args = parser.parse_args()

    args.cfg = '../experiments/DCMT_{}/Retrain_{}.yaml'.format(args.arch_type, args.arch_type)
    args.init_rect = [653, 221, 55, 40]  # ChasingDrones
    return args


def track(inputs):
    video_player = inputs['player']
    siam_tracker = inputs['tracker']
    siam_net = inputs['network']
    args = inputs['args']
    config = inputs['config']
    start_frame, lost, boxes, writer = 0, 0, [], None

    for count, im in enumerate(video_player):
        print('image idx: ', count)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        if count == start_frame:  # init
            # initialize video writer
            writer = cv2.VideoWriter('../video/' + args.video_name + '_' + args.arch_type + '_result' + '.mp4',
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     30,
                                     (im.shape[1], im.shape[0]))
            # initialize video tracker
            init_rect = args.init_rect
            cx = init_rect[0] + (init_rect[2] - 1) / 2  # center_x
            cy = init_rect[1] + (init_rect[3] - 1) / 2  # center_y
            w, h = init_rect[2], init_rect[3]
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            init_inputs = {'image': im, 'pos': target_pos, 'sz': target_sz, 'model': siam_net}
            siam_tracker.init(init_inputs)  # init tracker
            # write the first frame
            bbox = list(map(int, init_rect))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
            writer.write(im)

        elif count > start_frame:  # tracking
            state = siam_tracker.track(im)
            location = boxhelper.cxy_wh_2_rect(state['pos'], state['sz'])
            bbox = list(map(int, location))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
            writer.write(im)
    writer.release()
    # cv2.destroyAllWindows()


def main():
    print('===> load config <====')
    args = parse_args()
    if args.cfg is not None:
        config = edict(reader.load_yaml(args.cfg))
    else:
        raise Exception('Please set the config file for tracking test!')

    # create derived model of super model
    with open('../arch/arch_{}.pkl'.format(str(args.arch_type)), 'rb') as f:
        arch_dict = pickle.load(f)
    arch_config = arch_dict['arch']
    derivedNetwork = getattr(model_derived, '%s_Net' % config.MODEL.NAME.upper())
    der_Net = lambda net_config: derivedNetwork(net_config, config=config)
    derived_model = der_Net(arch_config)

    # create model
    print('====> build model <====')
    siam_net = DemoSiamInference(derived_model, config)

    # load checkpoint
    print('===> init Siamese <====')
    if args.resume is None or args.resume == 'None':
        resume = config.DEMO.RESUME
    else:
        resume = args.resume
    siam_net = loader.load_pretrain(siam_net, resume, print_unuse=False)
    siam_net.eval()
    siam_net = siam_net.cuda()

    # create tracker
    siam_tracker = tracker_builder.DemoTracker(config)

    print('===> init video player <====')
    if args.video is not None:
        video_name = args.video.split('/')[-1].split('.')[0]
        args.video_name = video_name
    video_player = player.get_frames(args.video)

    print('===> tracking! <====')
    inputs = {'player': video_player,
              'tracker': siam_tracker,
              'network': siam_net,
              'args': args,
              'config': config,
              }
    track(inputs)


if __name__ == '__main__':
    main()
