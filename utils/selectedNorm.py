import torch
from torch import nn as nn
from torch.nn import functional as F

# Norma = "GroupNorm"
Norma = "InstanceNorm2d"
# Norma = "BatchNorm2d"
# Norma = "DomainNorm"


class SelectedNorm(nn.Module):
    if Norma == "DomainNorm":

        def __init__(self, channel):
            super(SelectedNorm, self).__init__()
            self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
            self.affine = False
            if self.affine:
                self.weight = nn.Parameter(torch.ones(1, channel, 1, 1))
                self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))

        def forward(self, x):
            x = self.normalize(x)
            x = F.normalize(x, p=2, dim=1)
            if self.affine == True:
                x = x * self.weight + self.bias
            return x

    if Norma == "InstanceNorm2d":

        def __init__(self, channel):
            super(SelectedNorm, self).__init__()
            self.norm = nn.InstanceNorm2d(channel)

        def forward(self, x):
            return self.norm(x)

    elif Norma == "BatchNorm2d":

        def __init__(self, channel):
            super(SelectedNorm, self).__init__()
            self.norm = nn.BatchNorm2d(channel)

        def forward(self, x):
            return self.norm(x)

    elif Norma == "GroupNorm":

        def __init__(self, channel):
            super(SelectedNorm, self).__init__()
            self.group = 16
            self.norm = nn.GroupNorm(self.group, channel)

        def forward(self, x):
            return self.norm(x)