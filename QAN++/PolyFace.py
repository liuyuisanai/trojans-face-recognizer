# -*- coding: utf-8 -*-
"""
EfficientFacePolyNet
Any questions contact:
yuliu@ee.cuhk.edu.hk or liuyuisanai@gmail.com
Alarm: Do Not Distribute!
"""
import torch
import math
import numpy as np
import torch.nn as nn
import sys
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import interpolate
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import cv2

__all__ = ['MyModel']

BN_f = None
BN_q = None

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

class get_fc_E(nn.Module):
    def __init__(self, in_feature, in_h, in_w, out_feature):
        super(get_fc_E, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_feature, affine=True, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
        self.bn2 = nn.BatchNorm1d(out_feature, affine=False, eps=2e-5, momentum=0.9)
    def forward(self, x):
        output = {}
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output['feature_nobn'] = x
        x = self.bn2(x)
        output['feature'] = x
        return output

class get_fc_EMom(nn.Module):
    def __init__(self, in_feature, in_h, in_w, out_feature):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_feature, affine=True)
        # self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
        #TODO here momentum should be 0.1!
        self.bn2 = nn.BatchNorm1d(out_feature, affine=False, eps=2e-5, momentum=0.9, track_running_stats=True)
    def forward(self, x):
        output = {}
        x = self.bn1(x)
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output['feature_nobn'] = x
        x = torch.cat((x, x), dim=0)
        x = self.bn2(x)
        output['feature'] = x[:1, :]
        return output

def conv1x1(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)

def activation(act_type='prelu'):
    if act_type == 'prelu':
        act = nn.PReLU()
    else:
        act = nn.ReLU(inplace=True)
    return act

class BasicBlock_v3(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_type='prelu', use_se=False, use_checkpoint=False):
        super(BasicBlock_v3, self).__init__()
        m = OrderedDict()
        m['bn1'] = BN_q(inplanes)
        m['conv1'] = conv3x3(inplanes, planes, stride=1)
        m['bn2'] = BN_q(planes)
        m['act1'] = activation(act_type)
        m['conv2'] = conv3x3(planes, planes, stride=stride)
        m['bn3'] = BN_q(planes)
        self.group1 = nn.Sequential(m)

        self.use_checkpoint = use_checkpoint
        self.use_se = use_se
        if self.use_se:
            s = OrderedDict()
            s['conv1'] = nn.Conv2d(planes, planes // 16, kernel_size=1, stride=1, padding=0)
            s['act1'] = activation(act_type)
            s['conv2'] = nn.Conv2d(planes // 16, planes, kernel_size=1, stride=1, padding=0)
            s['act2'] = nn.Sigmoid()
            self.se_block = nn.Sequential(s)

        self.act2 = activation(act_type)
        self.downsample = downsample

        bypass_bn_weight_list.append(self.group1.bn3.weight)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        if self.use_checkpoint:
            out = checkpoint(self.group1, x)
        else:
            out = self.group1(x)

        if self.use_se:
            weight = F.adaptive_avg_pool2d(out, output_size=1)
            weight = self.se_block(weight)
            out = out * weight

        out = out + residual

        return self.act2(out)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn = BN_f(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Step1(nn.Module):

    def __init__(self):
        super(Step1,self).__init__()
        self.stem = nn.Sequential(
            BasicConv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch0 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)
        self.branch1 = BasicConv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        xs = self.stem(x)
        x0 = self.branch0(xs)
        x1 = self.branch1(xs)
        out = torch.cat((x0, x1), 1)
        return out

class Step2(nn.Module):

    def __init__(self):
        super(Step2, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 160, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = 0)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 160, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 64, out_channels = 64, kernel_size = (7,1), stride = (1,1), padding = (3,0)),
            BasicConv2d(in_channels = 64, out_channels = 64, kernel_size = (1,7), stride = (1,1), padding = (0,3)),
            BasicConv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = 0)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Step3(nn.Module):

    def __init__(self):
        super(Step3, self).__init__()
        self.branch0 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)
        self.branch1 = BasicConv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.step1 = Step1()
        self.step2 = Step2()
        self.step3 = Step3()

    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        return x


class BlockA(nn.Module):

    def __init__(self, use_checkpoint=False, keep_prob=0.8, multFlag=True):
        super(BlockA, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch2 = BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 384, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN_f(384)
        )
        self.prob = keep_prob
        self.multFlag = multFlag
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        a = torch.equal(self.m.sample(),torch.ones(1))
        identity = x.clone()
        if self.training:
            if a:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch2.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                if self.use_checkpoint:
                    x0 = checkpoint(self.branch0, x)
                    x1 = checkpoint(self.branch1, x)
                    x2 = checkpoint(self.branch2, x)
                    out = torch.cat((x0, x1, x2), 1)
                    out = checkpoint(self.stem, out)
                else:
                    x0 = self.branch0(x)
                    x1 = self.branch1(x)
                    x2 = self.branch2(x)
                    out = torch.cat((x0, x1, x2), 1)
                    out = self.stem(out)
                result = identity + 0.3*out
            else:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch2.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False
                result = identity
        else:
            if self.use_checkpoint:
                x0 = checkpoint(self.branch0, x)
                x1 = checkpoint(self.branch1, x)
                x2 = checkpoint(self.branch2, x)
                out = torch.cat((x0, x1, x2), 1)
                out = checkpoint(self.stem, out)
            else:
                x0 = self.branch0(x)
                x1 = self.branch1(x)
                x2 = self.branch2(x)
                out = torch.cat((x0, x1, x2), 1)
                out = self.stem(out)
            if self.multFlag:
                result = identity + 0.3 * out * self.prob
            else:
                retult = identity + 0.3 * out

        result = self.relu(result)
        return result

class BlockABranch(nn.Module):
    def __init__(self):
        super(BlockABranch, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch2 = BasicConv2d(in_channels = 384, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 384, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN_f(384)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.stem(out)
        return out

class BlockA2B(nn.Module):

    def __init__(self):
        super(BlockA2B, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 384, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch1 = BasicConv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 2, padding = 0)
        self.branch2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class BlockB(nn.Module):

    def __init__(self, use_checkpoint=False, keep_prob=0.8, multFlag=True):
        super(BlockB, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 128, out_channels = 160, kernel_size = (1,7), stride = (1,1), padding = (0,3)),
            BasicConv2d(in_channels = 160, out_channels = 192, kernel_size = (7,1), stride = (1,1), padding = (3,0))
        )
        self.branch1 = BasicConv2d(in_channels = 1152, out_channels = 192, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 1152, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN_f(1152)
        )
        self.prob = keep_prob
        self.multFlag = multFlag
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        a = torch.equal(self.m.sample(),torch.ones(1))
        identity = x.clone()
        if self.training:
            if a:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                if self.use_checkpoint:
                    x0 = checkpoint(self.branch0, x)
                    x1 = checkpoint(self.branch1, x)
                    out = torch.cat((x0, x1), 1)
                    out = checkpoint(self.stem, out)
                else:
                    x0 = self.branch0(x)
                    x1 = self.branch1(x)
                    out = torch.cat((x0, x1), 1)
                    out = self.stem(out)
                result = identity + 0.3*out
            else:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False
                result = identity
        else:
            if self.use_checkpoint:
                x0 = checkpoint(self.branch0, x)
                x1 = checkpoint(self.branch1, x)
                out = torch.cat((x0, x1), 1)
                out = checkpoint(self.stem, out)
            else:
                x0 = self.branch0(x)
                x1 = self.branch1(x)
                out = torch.cat((x0, x1), 1)
                out = self.stem(out)
            if self.multFlag:
                result = identity + 0.3 * out * self.prob
            else:
                retult = identity + 0.3 * out

        result = self.relu(result)
        return result

class BlockBBranch(nn.Module):
    def __init__(self):
        super(BlockBBranch, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 128, out_channels = 160, kernel_size = (1,7), stride = (1,1), padding = (0,3)),
            BasicConv2d(in_channels = 160, out_channels = 192, kernel_size = (7,1), stride = (1,1), padding = (3,0))
        )
        self.branch1 = BasicConv2d(in_channels = 1152, out_channels = 192, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 1152, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN_f(1152)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.stem(out)

        return out

class BlockB2C(nn.Module):

    def __init__(self):
        super(BlockB2C, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels = 1152, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out #2048

class BlockC(nn.Module):

    def __init__(self, use_checkpoint=False, keep_prob=0.8, multFlag=True):
        super(BlockC, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels = 2048, out_channels = 192, kernel_size = 1, stride = 1, padding = 0),
            BasicConv2d(in_channels = 192, out_channels = 224, kernel_size = (1,3), stride = (1,1), padding = (0,1)),
            BasicConv2d(in_channels = 224, out_channels = 256, kernel_size = (3,1), stride = (1,1), padding = (1,0))
        )
        self.branch1 = BasicConv2d(in_channels = 2048, out_channels = 192, kernel_size = 1, stride = 1, padding = 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 2048, kernel_size = 1, stride = 1, padding = 0, bias=False),
            BN_f(2048)
        )
        self.prob = keep_prob
        self.multFlag = multFlag
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        a = torch.equal(self.m.sample(),torch.ones(1))
        identity = x.clone()
        if self.training:
            if a:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = True

                if self.use_checkpoint:
                    x0 = checkpoint(self.branch0, x)
                    x1 = checkpoint(self.branch1, x)
                    out = torch.cat((x0, x1), 1)
                    out = checkpoint(self.stem, out)
                else:
                    x0 = self.branch0(x)
                    x1 = self.branch1(x)
                    out = torch.cat((x0, x1), 1)
                    out = self.stem(out)
                result = identity + 0.3*out
            else:
                for item in self.branch0.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.branch1.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False

                for item in self.stem.modules():
                    if isinstance(item, nn.Conv2d):
                        item.weight.requires_grad = False
                result = identity
        else:
            if self.use_checkpoint:
                x0 = checkpoint(self.branch0, x)
                x1 = checkpoint(self.branch1, x)
                out = torch.cat((x0, x1), 1)
                out = checkpoint(self.stem, out)
            else:
                x0 = self.branch0(x)
                x1 = self.branch1(x)
                out = torch.cat((x0, x1), 1)
                out = self.stem(out)
            if self.multFlag:
                result = identity + 0.3 * out * self.prob
            else:
                retult = identity + 0.3 * out

        result = self.relu(result)
        return result

class APolynet(nn.Module):

    def __init__(self, feature_dim, bn_mom=0.1, bn_eps=1e-10, fc_type='E',
                 num_blocks=[10, 20, 10],
                 checkpoints=[0, 0, 0],
                 att_mode='none'):
        super(APolynet, self).__init__()

        self.att_mode = att_mode

        global BN_f
        def BNFunc(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, momentum=bn_mom, eps=bn_eps)
        BN_f = BNFunc

        self.stem = Stem()

        self.a2b = BlockA2B()
        self.b2c = BlockB2C()

        if self.att_mode == 'none':
            self.a10 = self._make_layer(BlockA, num_blocks[0], checkpoints=checkpoints[0])
            self.b20 = self._make_layer(BlockB, num_blocks[1], checkpoints=checkpoints[1])
            self.c10 = self._make_layer(BlockC, num_blocks[2], checkpoints=checkpoints[2])
        else:
            raise RuntimeError('unknown att_mode: {}'.format(self.att_mode))

        self.fc = get_fc_E(2048, 6, 6, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                scale = math.sqrt(3. / fan_in)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _make_layer(self, block, num_blocks, checkpoints=0):
        layers = []
        for i in range(num_blocks):
            layers.append(block(use_checkpoint=checkpoints>i))

        return nn.Sequential(*layers)

    def forward(self, x, flip=False):
        output = {}

        img_list = []
        for cnt in range(x.size(0)):
            tmp = x[cnt]
            tmp = tmp.cpu().numpy()
            tmp = tmp.astype(np.uint8)
            tmp = tmp.transpose((1, 2, 0))
            tmp = cv2.resize(tmp[1:, 1:, :], (235, 235))
            tmp = tmp*3.2/255.0 - 1.6
            if flip:
                tmp = cv2.flip(tmp, 1)
            tmp = tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp)
            tmp = tmp.float()
            tmp = tmp[None, ...]
            img_list.append(tmp)
        x = torch.cat(img_list, 0)
        x = x.cuda()

        ori = x
        x = self.stem(x)

        if self.att_mode == 'none':
            x = self.a10(x)
            x = self.a2b(x)
            x = self.b20(x)
            x = self.b2c(x)
            x = self.c10(x)
        else:
            raise RuntimeError('unknown att_mode: {}'.format(self.att_mode))

        headout= self.fc(x)
        output.update(headout)
        return output

class Quality(nn.Module):
    '''
    Designed for 112x112 input, keep higher feature map:
    the first conv is set to be (kernel=3, stride=1),
    remove maxpooling in the first stage.
    fewer channels [64, 128, 256, 512].
    '''
    def __init__(self, block, layers, feature_dim, final_reduction_dim=0, channels=[64,128,256,512], checkpoints=[0,0,0,0],
                 ratios=[1,1,1,1], groups=[1,1,1,1],
                 act_type='prelu', fc_type='EMom', use_se=False,
                 bn_mom=0.1, bn_eps=2e-5, bypass_last_bn=False,
                 att_mode='none'):

        super(Quality, self).__init__()
        self.act_type = act_type
        self.inplanes = channels[0]
        self.feature_dim = feature_dim
        self.checkpoints = checkpoints
        self.use_se = use_se
        self.att_mode = att_mode
        self.final_reduction_dim = final_reduction_dim

        global BN_q, bypass_bn_weight_list
        def BNFunc(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, momentum=bn_mom, eps=bn_eps)
        BN_q = BNFunc

        bypass_bn_weight_list = []

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        m['bn1'] = BN_q(self.inplanes)
        m['act1'] = activation(act_type)
        self.group1 = nn.Sequential(m)

        if self.att_mode == 'none':
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2, checkpoints=checkpoints[0], ratio=ratios[0], groups=groups[0])
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, checkpoints=checkpoints[1], ratio=ratios[1], groups=groups[1])
            self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, checkpoints=checkpoints[2], ratio=ratios[2], groups=groups[2])
            self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, checkpoints=checkpoints[3], ratio=ratios[3], groups=groups[3])

        if self.final_reduction_dim > 0:
            m = OrderedDict()
            m['bn1'] = BN_q(channels[3]*block.expansion)
            m['conv1'] = conv1x1(channels[3]*block.expansion, self.final_reduction_dim, stride=1)
            self.final_reduction = nn.Sequential(m)
            self.fc1 = head.__dict__['get_fc_'+fc_type](self.final_reduction_dim, 7, 7, feature_dim)
        else:
            self.fc1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels[3]*block.expansion,1,kernel_size=1,stride=1,padding=0,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_normal(m.weight.data)
                fan_in = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                scale = math.sqrt(2. / fan_in)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            # Xavier can not be applied to less than 2D.
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.uniform_(-scale, scale)

        if bypass_last_bn:
            assert block is BasicBlock_v3
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in {}'.format(len(bypass_bn_weight_list), block))


    def _make_layer(self, block, planes, blocks, stride=1, checkpoints=0, ratio=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN_q(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se,
                                act_type=self.act_type, use_checkpoint=checkpoints>0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se,
                                    act_type=self.act_type, use_checkpoint=checkpoints>i))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = {''}

        img_list = []
        # import pdb; pdb.set_trace()
        for cnt in range(x.size(0)):
            tmp = x[cnt]
            tmp = tmp.cpu().numpy()
            tmp = tmp.astype(np.uint8)
            tmp = tmp.transpose((1, 2, 0))
            tmp = cv2.resize(tmp[1:, 1:, :], (112, 112))
            tmp = tmp*3.2/255.0 - 1.6
            tmp = tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp)
            tmp = tmp.float()
            tmp = tmp[None, ...]
            img_list.append(tmp)
        x = torch.cat(img_list, 0)
        x = x.cuda()

        ori = x
        # import pdb; pdb.set_trace()
        x = self.group1(x)

        if self.att_mode == 'none':
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            raise RuntimeError('unknown att_mode: {}'.format(self.att_mode))

        if self.final_reduction_dim > 0:
            x = self.final_reduction(x)
        headout = self.fc1(x)
        output = {'quality':headout}

        return output

class MyModel(object):
    def __init__(self, feature_dim, ckpt_file):
        model_feature = APolynet(feature_dim=feature_dim, num_blocks=[20, 30, 20])
        model_quality = Quality(BasicBlock_v3, [2, 2, 2, 2], feature_dim=feature_dim, channels=[8,16,32,48])
        self.model_feature = model_feature.cuda()
        self.model_quality = model_quality.cuda()

        cudnn.benchmark = True

        def map_func_cuda(storage, location):
            return storage.cuda()

        ckpt = torch.load(ckpt_file, map_location=map_func_cuda)
        state_dict = remove_prefix(ckpt['ckpt_quality'], 'module.base.')
        self.model_quality.load_state_dict(state_dict, strict=True)

        # ckpt = torch.load(quality_ckpt, map_location=map_func_cuda)
        state_dict = remove_prefix(ckpt['ckpt_feature'], 'module.base.')
        self.model_feature.load_state_dict(state_dict, strict=False)

    def eval(self):
        self.model_feature.eval()
        self.model_quality.eval()

    def __call__(self, input_data):
        # quality = self.model_quality(input_data['image_small'].cuda())
        # feature = self.model_feature(input_data['image'].cuda())
        # feature_flip = self.model_feature(input_data['image_flip'].cuda())
        quality = self.model_quality(input_data.cuda())
        feature = self.model_feature(input_data.cuda())
        feature_flip = self.model_feature(input_data.cuda(), flip=True)

        final_feature = (feature['feature_nobn'].data.cpu().numpy() + \
                         feature_flip['feature_nobn'].data.cpu().numpy()) / 2.0
        final_quality = quality['quality'].data.cpu().numpy()
        print(final_quality)
        output = {'feature': final_feature, \
                  'quality': final_quality}

        return output
