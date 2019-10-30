import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class ArcNegFace(nn.Module):
    def __init__(self, in_features, out_features, scale=64, margin=0.5, easy_margin=False):
        super(ArcNegFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.alpha = 1.2
        self.sigma = 2
        self.thresh = math.cos(math.pi-self.margin)
        self.mm = math.sin(math.pi-self.margin) * self.margin
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())

        a = torch.zeros_like(cos)
        if self.easy_margin:
            for i in range(a.size(0)):
                lb = int(label[i])
                if cos[i, lb].data[0] > 0:
                    a[i, lb] = a[i, lb] + self.margin
            return self.scale * torch.cos(torch.acos(cos) + a)
        else:
            b = torch.zeros_like(cos)
            a_scale = torch.zeros_like(cos)
            c_scale = torch.ones_like(cos)
            t_scale = torch.ones_like(cos)
            for i in range(a.size(0)):
                lb = int(label[i])
                a_scale[i,lb]=1
                c_scale[i,lb]=0
                if cos[i, lb].item() > self.thresh:
                    a[i, lb] = torch.cos(torch.acos(cos[i, lb])+self.margin)
                else:
                    a[i, lb] = cos[i, lb]-self.mm
                reweight = self.alpha*torch.exp(-torch.pow(cos[i,]-a[i,lb].item(),2)/self.sigma)
                t_scale[i]*=reweight.detach()
            return {'logits':self.scale * (a_scale*a+c_scale*(t_scale*cos+t_scale-1))}#,{'pos_scale':a_scale.cpu().data.numpy(), 'neg_scale':c_scale.cpu().data.numpy(), 'cos':cos.cpu().data.numpy()}


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale=64, margin=0.5, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        self.thresh = math.cos(math.pi-self.margin)
        self.mm = math.sin(math.pi-self.margin) * self.margin
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label, curr_step=-1):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())
        a = torch.zeros_like(cos)
        thetas = []

        if self.easy_margin == -1:
            for i in range(a.size(0)):
                lb = int(label[i])
                a[i, lb] = a[i, lb] + self.margin
                theta = math.acos(cos[i, lb].data[0]) / math.pi * 180
                thetas.append(theta)
                return {'logits': self.scale * torch.cos(torch.acos(cos) + a)}
        elif self.easy_margin is True:
            for i in range(a.size(0)):
                lb = int(label[i])
                if cos[i, lb].data[0] > 0:
                    a[i, lb] = a[i, lb] + self.margin
                theta = math.acos(cos[i, lb].data[0]) / math.pi * 180
                thetas.append(theta)
                return {'logits': self.scale * torch.cos(torch.acos(cos) + a)}
        else:
            b = torch.zeros_like(cos)

            for i in range(a.size(0)):
                lb = int(label[i])
                theta = math.acos(cos[i, lb].item()) / math.pi * 180
                thetas.append(theta)
                if cos[i, lb].item() > self.thresh:
                    a[i, lb] = a[i, lb] + self.margin
                else:
                    b[i, lb] = b[i, lb] - self.mm
            return {'logits': self.scale * ( torch.cos(torch.acos(cos) + a) + b )}

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', scale=' + str(self.scale) \
            + ', margin=' + str(self.margin) + ')'
