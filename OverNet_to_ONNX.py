import json
import importlib
import argparse
import os

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.onnx
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default='OverNet')
    parser.add_argument("--ckpt_path", type=str, default='./checkpoints/OverNet_test_x2.pth.tar')
    parser.add_argument('--image', type=str, required=True, help='input image to use')
    parser.add_argument('--model_out', type=str, default='OverNet.onnx')
    parser.add_argument("--ONNX_dir", type=str, default='ONNX')
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--upscale", type=int, default=3)
    return parser.parse_args()


def main(cfg):

    module = importlib.import_module("{}".format(cfg.model))
    net = module.Network(scale=cfg.scale,upscale=cfg.upscale,group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: " + str(device))

    state_dict = torch.load(cfg.ckpt_path, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])
    print("Model is loaded...")

    img = Image.open(cfg.image)
    img_to_tensor = ToTensor()
    input = img_to_tensor(img).view(1,-1,img.size[1], img.size[0]).to(device)
    print('Input image size ---> {:d}x{:d}'.format(img.size[0], img.size[1]))

    output = ['SR']
    dynamic_axes= {'inputs':{0:'input' , 2:'scale', 3:'upscale'}}

    print('Exporting model to ONNX...')

    ONNX_dir = os.path.join(cfg.ONNX_dir,"x{}".format(cfg.scale))
    os.makedirs(ONNX_dir, exist_ok=True)
    file_name = os.path.join(ONNX_dir, "OverNet.onnx")

    torch.onnx.export(net, (input, cfg.scale, cfg.upscale), file_name, export_params=True, opset_version=13,
                            input_names = ['inputs','scale','upscale'], output_names=output, dynamic_axes=dynamic_axes)

    print('Model exported to {:s}'.format(cfg.model_out))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
