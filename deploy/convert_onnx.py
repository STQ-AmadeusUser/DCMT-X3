import _init_paths
import io
import numpy as np
from torch import nn
from easydict import EasyDict as edict
import torch.onnx
import argparse
import pickle
import utils.read_file as reader
import utils.model_helper as loader
import search.model_derived as model_derived
from models.deployinference import DeployInference


batch_size = 1


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Retrain DCMT')
    parser.add_argument('--arch_type', type=str, default='lt', help='retrain DCMT from which type')
    args = parser.parse_args()
    args.cfg = '../experiments/DCMT_{}/Retrain_{}.yaml'.format(args.arch_type, args.arch_type)

    # TODO: change it to the checkpoint you need
    if args.arch_type == 'res':
        args.resume = '../snapshot/DCMT_ResNet_GOT10KTEST.pth'
    elif args.arch_type == 'lt':
        args.resume = '../snapshot/DCMT_LightTrack_GOT10KTEST.pth'
    elif args.arch_type == 'al':
        args.resume = '../snapshot/DCMT_AlexNet_UAV10FPS.pth'

    return args


# reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
if __name__ == '__main__':
    # load config
    args = parse_args()
    config = edict(reader.load_yaml(args.cfg))

    # build DCMT network for tracking
    with open('../arch/arch_{}.pkl'.format(args.arch_type), 'rb') as f:
        arch_dict = pickle.load(f)
    arch_config = arch_dict['arch']
    derivedNetwork = getattr(model_derived, '%s_Net' % config.MODEL.NAME.upper())
    der_Net = lambda net_config: derivedNetwork(net_config, config=config)
    derived_model = der_Net(arch_config)
    model = DeployInference(derived_model, config)

    # load pretrain
    model = loader.load_pretrain(model, args.resume, print_unuse=False).eval().to('cpu')

    # Input to the model
    torch_z = torch.randn(batch_size, 3, config.TRAIN.TEMPLATE_SIZE, config.TRAIN.TEMPLATE_SIZE, requires_grad=True)
    torch_x = torch.randn(batch_size, 3, config.TRAIN.SEARCH_SIZE, config.TRAIN.SEARCH_SIZE, requires_grad=True)
    torch_b = torch.randn(batch_size, 4, 1, 1, requires_grad=True)
    torch_cls, torch_reg = model(torch_z, torch_x, torch_b)

    # Export the model
    torch.onnx.export(model,  # model being run
                      (torch_z, torch_x, torch_b),  # model input (or a tuple for multiple inputs)
                      "DCMT_{}.onnx".format(args.arch_type),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input1', 'input2', 'input3'],  # the model's input names
                      output_names=['output1', 'output2'],  # the model's output names
                      )
