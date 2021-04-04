import torch.nn as nn
from torchvision.models import mobilenet as vmn
import torch.utils.model_zoo as model_zoo

class MobileNet(vmn.MobileNetV2):
    'MobileNetV2: Inverted Residuals and Linear Bottlenecks - https://arxiv.org/abs/1801.04381'

    def __init__(self, outputs=[18], url=None):
        self.stride = 128
        self.url = url
        super().__init__()
        self.outputs = outputs
        self.unused_modules = ['features.18', 'classifier']

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        outputs = []
        for indx, feat in enumerate(self.features[:-1]):
            x = feat(x)
            if indx in self.outputs:
                outputs.append(x)
        return outputs
