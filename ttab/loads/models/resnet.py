# -*- coding: utf-8 -*-
import math
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp
from timm.models._manipulate import checkpoint_seq as timm_checkpoint_seq
from ttab.configs.datasets import dataset_defaults

__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


class ViewFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(group_norm_num_groups, out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm2d(group_norm_num_groups, out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm2d(group_norm_num_groups, out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        return self.relu(out)


class ResNetBase(nn.Module):
    def _init_conv(self, m):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def _init_bn(self, m):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    def _init_fc(self, m):
        m.weight.data.normal_(mean=0, std=0.01)
        if m.bias is not None:
            m.bias.data.zero_()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d):
                self._init_bn(m)
            elif isinstance(m, nn.Linear):
                self._init_fc(m)

    def _make_block(self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_fn.expansion, kernel_size=1, stride=stride, bias=False),
                norm2d(group_norm_num_groups, planes * block_fn.expansion),
            )
        layers = [block_fn(self.inplanes, planes, stride, downsample, group_norm_num_groups)]
        self.inplanes = planes * block_fn.expansion
        for _ in range(1, block_num):
            layers.append(block_fn(self.inplanes, planes, group_norm_num_groups=group_norm_num_groups))
        return nn.Sequential(*layers)


def checkpoint_seq(functions, x, every=1, flatten=False, skip_last=False):
    """Wrapper over timm_checkpoint_seq without preserve_rng_state."""
    return timm_checkpoint_seq(functions, x, every=every, flatten=flatten, skip_last=skip_last)


class ResNetImagenet(ResNetBase):
    def __init__(self, num_classes, depth, split_point='layer4', group_norm_num_groups=None, grad_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        if split_point not in ['layer3', 'layer4', None]:
            raise ValueError(f"invalid split_point={split_point}")
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint
        params = {18: (BasicBlock, [2,2,2,2]), 34: (BasicBlock, [3,4,6,3]),
                  50: (Bottleneck, [3,4,6,3]), 101: (Bottleneck, [3,4,23,3]), 152: (Bottleneck, [3,8,36,3])}
        block_fn, layers_cfg = params[depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm2d(group_norm_num_groups, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_block(block_fn, 64, layers_cfg[0], group_norm_num_groups=group_norm_num_groups)
        self.layer2 = self._make_block(block_fn, 128, layers_cfg[1], stride=2, group_norm_num_groups=group_norm_num_groups)
        self.layer3 = self._make_block(block_fn, 256, layers_cfg[2], stride=2, group_norm_num_groups=group_norm_num_groups)
        self.layer4 = self._make_block(block_fn, 512, layers_cfg[3], stride=2, group_norm_num_groups=group_norm_num_groups)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(512 * block_fn.expansion, num_classes, bias=False)
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x)
            x = checkpoint_seq(self.layer2, x)
            x = checkpoint_seq(self.layer3, x)
            if self.split_point in ['layer4', None]:
                x = checkpoint_seq(self.layer4, x)
        else:
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
            if self.split_point in ['layer4', None]: x = self.layer4(x)
        if self.split_point in ['layer4', None]:
            x = self.avgpool(x); x = x.view(x.size(0), -1)
        return x

    def forward_head(self, x, pre_logits=False):
        if self.split_point == 'layer3':
            if self.grad_checkpoint:
                x = checkpoint_seq(self.layer4, x)
            else:
                x = self.layer4(x)
            x = self.avgpool(x); x = x.view(x.size(0), -1)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)


class ResNetCifar(ResNetBase):
    def __init__(self, num_classes, depth, split_point='layer3', group_norm_num_groups=None, grad_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        if split_point not in ['layer2', 'layer3', None]:
            raise ValueError(f"invalid split_point={split_point}")
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint
        if depth % 6 != 2:
            raise ValueError('depth must be 6n+2')
        block_nums = (depth - 2) // 6
        block_fn = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm2d(group_norm_num_groups, 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_block(block_fn, 16, block_nums, group_norm_num_groups=group_norm_num_groups)
        self.layer2 = self._make_block(block_fn, 32, block_nums, stride=2, group_norm_num_groups=group_norm_num_groups)
        self.layer3 = self._make_block(block_fn, 64, block_nums, stride=2, group_norm_num_groups=group_norm_num_groups)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * block_fn.expansion, num_classes, bias=False)
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x)
            x = checkpoint_seq(self.layer2, x)
            if self.split_point in ['layer3', None]:
                x = checkpoint_seq(self.layer3, x)
                x = self.avgpool(x); x = x.view(x.size(0), -1)
        else:
            x = self.layer1(x); x = self.layer2(x)
            if self.split_point in ['layer3', None]: x = self.layer3(x); x = self.avgpool(x); x = x.view(x.size(0), -1)
        return x

    def forward_head(self, x, pre_logits=False):
        if self.split_point == 'layer2':
            if self.grad_checkpoint: x = checkpoint_seq(self.layer3, x)
            else: x = self.layer3(x)
            x = self.avgpool(x); x = x.view(x.size(0), -1)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)


class ResNetAffectNet(ResNetBase):
    def __init__(self, num_classes, depth=50, split_point='layer4', group_norm_num_groups=None, grad_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        if split_point not in ['layer3', 'layer4', None]:
            raise ValueError(f"invalid split_point={split_point}")
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint
        if depth in [18,34]: block_fn, cfg = BasicBlock, {18:[2,2,2,2],34:[3,4,6,3]}[depth]
        elif depth in [50,101,152]: block_fn, cfg = Bottleneck, {50:[3,4,6,3],101:[3,4,23,3],152:[3,8,36,3]}[depth]
        else: raise ValueError('unsupported depth')
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1 = norm2d(group_norm_num_groups,64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_block(block_fn,64,cfg[0],group_norm_num_groups=group_norm_num_groups)
        self.layer2 = self._make_block(block_fn,128,cfg[1],stride=2,group_norm_num_groups=group_norm_num_groups)
        self.layer3 = self._make_block(block_fn,256,cfg[2],stride=2,group_norm_num_groups=group_norm_num_groups)
        self.layer4 = self._make_block(block_fn,512,cfg[3],stride=2,group_norm_num_groups=group_norm_num_groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512*block_fn.expansion,num_classes,bias=False)
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x)
            x = checkpoint_seq(self.layer2, x)
            x = checkpoint_seq(self.layer3, x)
            if self.split_point in ['layer4', None]: x = checkpoint_seq(self.layer4, x)
        else:
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
            if self.split_point in ['layer4', None]: x = self.layer4(x)
        x = self.avgpool(x); x = x.view(x.size(0), -1)
        return x

    def forward_head(self, x, pre_logits=False):
        if self.split_point == 'layer3':
            if self.grad_checkpoint: x = checkpoint_seq(self.layer4, x)
            else: x = self.layer4(x)
            x = self.avgpool(x); x = x.view(x.size(0), -1)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class ResNetMNIST(ResNetBase):
    def __init__(self, num_classes, depth, in_dim, split_point='layer4', group_norm_num_groups=None):
        super().__init__()
        self.num_classes = num_classes
        if split_point not in ['layer3','layer4', None]: raise ValueError('invalid split_point')
        self.split_point = split_point
        params = {18:(BasicBlock,[2,2,2,2]),34:(BasicBlock,[3,4,6,3]),50:(Bottleneck,[3,4,6,3]),101:(Bottleneck,[3,4,23,3]),152:(Bottleneck,[3,8,36,3])}
        block_fn, layers_cfg = params[depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_dim,64,7,2,3,bias=False)
        self.bn1 = norm2d(group_norm_num_groups,64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_block(block_fn,64,layers_cfg[0],group_norm_num_groups=group_norm_num_groups)
        self.layer2 = self._make_block(block_fn,128,layers_cfg[1],stride=2,group_norm_num_groups=group_norm_num_groups)
        self.layer3 = self._make_block(block_fn,256,layers_cfg[2],stride=2,group_norm_num_groups=group_norm_num_groups)
        self.layer4 = self._make_block(block_fn,512,layers_cfg[3],stride=2,group_norm_num_groups=group_norm_num_groups)
        self.classifier = nn.Linear(512*block_fn.expansion,num_classes,bias=False)
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        if self.split_point in ['layer4', None]:
            x = self.layer4(x); x = x.view(x.size(0), -1)
        return x

    def forward_head(self, x, pre_logits=False):
        if self.split_point == 'layer3':
            x = self.layer4(x); x = x.view(x.size(0), -1)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def resnet(dataset, depth, split_point=None, group_norm_num_groups=None, grad_checkpoint=False):
    num_classes = dataset_defaults[dataset]['statistics']['n_classes']
    if 'mnist' in dataset:
        in_dim = 1 if dataset=='mnist' else 3
        return ResNetMNIST(num_classes, depth, in_dim, split_point, group_norm_num_groups)
    elif 'yearbook' in dataset:
        return ResNetMNIST(num_classes, depth, in_dim=3, split_point=split_point, group_norm_num_groups=group_norm_num_groups)
    elif 'cifar' in dataset:
        return ResNetCifar(num_classes, depth, split_point, group_norm_num_groups, grad_checkpoint)
    elif 'affectnet' in dataset:
        return ResNetAffectNet(num_classes, depth, split_point, group_norm_num_groups, grad_checkpoint)
    else:
        return ResNetImagenet(num_classes, depth, split_point, group_norm_num_groups, grad_checkpoint)
