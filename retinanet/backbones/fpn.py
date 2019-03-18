import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet as vrn

from .resnet import ResNet
from .utils import register

class FPN(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        is_light = features.bottleneck == vrn.BasicBlock
        channels = [128, 256, 512] if is_light else [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return [p3, p4, p5, p6, p7]

@register
def ResNet18FPN():
    return FPN(ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet18']))

@register
def ResNet34FPN():
    return FPN(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet34']))

@register
def ResNet50FPN():
    return FPN(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet50']))

@register
def ResNet101FPN():
    return FPN(ResNet(layers=[3, 4, 23, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet101']))

@register
def ResNet152FPN():
    return FPN(ResNet(layers=[3, 8, 36, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet152']))

