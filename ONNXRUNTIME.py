import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import onnxruntime
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='OverNet')
    parser.add_argument("--output_dir", type=str, required=True, help='Please write the output direction')
    parser.add_argument("--test_data_dir", type=str, required=True, help='Input images to use')
    parser.add_argument("--scale", type=int, default=2)

    return parser.parse_args()

def save_image(tensor, filename):
    tensor = torch.tensor(tensor[0]).squeeze(0).cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def main(cfg):

    img_to_tensor = ToTensor()

    ort_session = onnxruntime.InferenceSession('./ONNX/x2/OverNet.onnx')

    all_imgs = glob.glob(os.path.join(cfg.test_data_dir, "*.png"))

    for i, img in enumerate(all_imgs):

        input = Image.open(img)
        input = img_to_tensor(input).view(1,-1,input.size[1], input.size[0])

        output = ort_session.run(None, {'inputs': input.numpy(),'scale': np.array(cfg.scale).astype(np.int64)})

        output_dir = os.path.join(cfg.output_dir, "x{}".format(cfg.scale))
        os.makedirs(output_dir, exist_ok=True)

        sr_im_path = os.path.join(output_dir, "{}_SR_.png".format(img.split(".")[0].split("/")[-1]))

        save_image(output, sr_im_path)
        print("Image_{} is upsampled to scale x{}".format(i,cfg.scale))
    print("Done!!")

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
