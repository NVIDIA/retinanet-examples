#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import torchvision
from torchvision.models import resnet as vrn
import torch.utils.model_zoo as model_zoo

from .utils import register

class ResNet(vrn.ResNet):
    'Deep Residual Network - https://arxiv.org/abs/1512.03385'

    def __init__(self, layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[5], groups=1, width_per_group=64, url=None):
        self.stride = 128        
        self.bottleneck = bottleneck
        self.outputs = outputs
        self.url = url

        kwargs = {'block': bottleneck, 'layers': layers, 'groups': groups, 'width_per_group': width_per_group}
        super().__init__(**kwargs)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs

@register
def ResNet18C4():
    return ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[4], url=vrn.model_urls['resnet18'])

@register
def ResNet34C4():
    return ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.BasicBlock, outputs=[4], url=vrn.model_urls['resnet34'])
