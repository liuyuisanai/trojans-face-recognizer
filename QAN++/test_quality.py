import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import cv2
import numpy as np

from apolynet_stodepth import MyModel

parser = argparse.ArgumentParser(description='Inference for iqiyi')
parser.add_argument('--test_img', type=str, help='test image path')

def set_input_old(img_path):
    img_ori = cv2.imread(img_path)
    img = cv2.resize(img_ori[1:, 1:, :], (235, 235))
    img = img*3.2/255.0 - 1.6
    img_flip = cv2.flip(img, 1)

    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    img = img.float()
    img = img[None, ...]
    img = img

    img_flip = img_flip.transpose((2,0,1))
    img_flip = torch.from_numpy(img_flip)
    img_flip = img_flip.float()
    img_flip = img_flip[None, ...]

    img_small = cv2.resize(img_ori[1:, 1:, :], (112, 112))
    img_small = img_small*3.2/255.0 - 1.6
    img_small = img_small.transpose((2,0,1))
    img_small = torch.from_numpy(img_small)
    img_small = img_small.float()
    img_small = img_small[None, ...]

    output = {'image': img, 'image_flip': img_flip, 'image_small': img_small}
    return output

def set_input(img_path):
    img = cv2.imread(img_path)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.float()
    img = img[None, ...]
    return img


def main():
    args = parser.parse_args()

    # load state
    model = MyModel(feature_dim=256, ckpt_file='qan.pth.tar')
    model.eval()

    with torch.no_grad():
        input_data = set_input(args.test_img)
        output = model(input_data)

    if not osp.exists('quality'):
        os.mkdir('quality')
    if not osp.exists('feature'):
        os.mkdir('feature')
    quality_name = osp.join('quality', osp.basename(args.test_img).replace('jpg', 'txt'))
    feature_name = osp.join('feature', osp.basename(args.test_img).replace('jpg', 'txt'))
    print('quality saved to: ', quality_name)
    print('feature saved to: ', feature_name)
    output['quality'].tofile(quality_name)
    output['feature'].tofile(feature_name)

if __name__ == '__main__':
    main()
