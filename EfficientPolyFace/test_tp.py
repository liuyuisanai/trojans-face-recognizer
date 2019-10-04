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

from EfficientPolyFace import apolynet_stodepth_deeper

parser = argparse.ArgumentParser(description='Inference for tp')
parser.add_argument('--test_img', type=str, help='test image path')
parser.add_argument('--save_path', type=str, default='feature', help='feature save path')

def set_input_old(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img[1:, 1:, :], (235, 235))
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

    output = {'image': img, 'image_flip': img_flip}
    return output

def set_input(img_path):
    img = cv2.imread(img_path)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.float()
    img = img[None, ...]
    return img

class MyDataset(Dataset):
    def __init__(self, prefix, list_file, crop_size, final_size, \
                 crop_center_y_offset=0, scale_aug=0.0, trans_aug=0.0, flip=-1, resize=-1):
        self.prefix = prefix
        with open(list_file) as f:
            list_lines = f.readlines()
            list_lines = [x.strip() for x in list_lines]
        self.num = len(list_lines)
        self.list_lines = list_lines

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        out = set_input(self.prefix + '/' + self.list_lines[idx])
        return out


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def main():
    args = parser.parse_args()

    # prepare model
    model = apolynet_stodepth_deeper(256)
    model = model.cuda()
    cudnn.benchmark = True

    # load state
    ckpt_path = 'ckpt.pth.tar'
    # import pdb; pdb.set_trace()
    def map_func_cuda(storage, location):
        return storage.cuda()
    ckpt = torch.load(ckpt_path, map_location=map_func_cuda)
    state_dict = remove_prefix(ckpt['state_dict'], 'module.base.')
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    test_feature = []
    with torch.no_grad():
        input_data = set_input(args.test_img)
        feature = model(input_data)
        feature_flip = model(input_data, flip=True)
        output_feature = (feature['feature_nobn'] + feature_flip['feature_nobn']) / 2.0

    if not osp.exists(args.save_path):
        os.mkdir(args.save_path)
    save_name = osp.join(args.save_path, osp.basename(args.test_img).replace('jpg', 'txt'))
    output_feature = output_feature.data.cpu().numpy()
    output_feature.tofile(save_name)


if __name__ == '__main__':
    main()
