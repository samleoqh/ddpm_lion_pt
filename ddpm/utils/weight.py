import torch
from torch import nn


def weights_init(var_scale=0.02):
    def _(m):
        with torch.no_grad():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1 or classname.find("ConvTranspose2d") != -1:
                nn.init.normal_(m.weight.data, 0.0, var_scale)
                # nn.init.xavier_normal_(m.weight.data, var_scale)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, var_scale)
                nn.init.constant_(m.bias.data, 0)
    return _