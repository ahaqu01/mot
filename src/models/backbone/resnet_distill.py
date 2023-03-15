# encoding: utf-8
import math
import torch.nn.functional as F
from torch import nn

from .layers.batch_norm import IBN, get_norm
from .layers.se_layer import SELayer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.channel_nums = []
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)

        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        self.channel_nums.append(self.inplanes)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(x, inplace=True)

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error!")
        return [bn1, bn2, bn3, bn4]

    def extract_feature(self, x, preReLU=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        return [feat1, feat2, feat3, feat4], F.relu(feat4)

    def get_channel_nums(self):
        return self.channel_nums


def build_resnet_backbone_distill(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    last_stride = cfg["MODEL"]["BACKBONE"]["LAST_STRIDE"]
    bn_norm = cfg["MODEL"]["BACKBONE"]["NORM"]
    with_ibn = cfg["MODEL"]["BACKBONE"]["WITH_IBN"]
    with_se = cfg["MODEL"]["BACKBONE"]["WITH_SE"]
    with_nl = cfg["MODEL"]["BACKBONE"]["WITH_NL"]
    depth = cfg["MODEL"]["BACKBONE"]["DEPTH"]
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    return model
