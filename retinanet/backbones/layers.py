import torch
from torch import nn
import torch.nn.functional as F

class FixedBatchNorm2d(nn.Module):
    'BatchNorm2d where the batch statistics and the affine parameters are fixed'

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        
    def forward(self, x):
        return F.batch_norm(x, running_mean=self.running_mean, running_var=self.running_var, weight=self.weight, bias=self.bias)

def convert_fixedbn_model(module):
    'Convert batch norm layers to fixed'

    mod = module
    if isinstance(module, nn.BatchNorm2d):
        mod = FixedBatchNorm2d(module.num_features)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_fixedbn_model(child))

    return mod
