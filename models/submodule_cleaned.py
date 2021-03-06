from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

import sys

sys.path.append("../")
from utils.activations_autofn import MishAuto

Act = nn.ReLU
# Act = SwishAuto
# Act = MishAuto

# Norm = nn.BatchNorm2d
Norm = nn.InstanceNorm2d
# Norm = nn.GroupNorm


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False,
        ),
        Norm(out_planes),
    )


class BasicBlock(nn.Module):
    def __init__(self, Cin, Cout, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(Cin, Cout, 3, stride, pad, dilation), Act(inplace=True)
        )

        self.conv2 = convbn(Cout, Cout, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(
                left,
                3,
                Variable(torch.LongTensor([i for i in range(shift, width)])).cuda(),
            ),
            (shift, 0, 0, 0),
        )
        shifted_right = F.pad(
            torch.index_select(
                right,
                3,
                Variable(torch.LongTensor([i for i in range(width - shift)])).cuda(),
            ),
            (shift, 0, 0, 0),
        )
        out = torch.cat((shifted_left, shifted_right), 1).view(
            batch, filters * 2, 1, height, width
        )
        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(
            torch.Tensor(
                np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])
            ).cuda(),
            requires_grad=False,
        )

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        data = x * disp
        out = torch.sum(x * disp, 1)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),
            Act(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            Act(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            Act(inplace=True),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(
                Cin=self.inplanes,
                Cout=self.inplanes,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes,
                Cout=self.inplanes,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes,
                Cout=self.inplanes,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(
                self.inplanes, self.inplanes * 2, kernel_size=1, stride=2, bias=False
            ),
            Norm(self.inplanes * 2),
        )

        self.layer2 = nn.Sequential(
            BasicBlock(
                Cin=self.inplanes,
                Cout=self.inplanes * 2,
                stride=2,
                downsample=self.downsample2,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 2,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(
                self.inplanes * 2,
                self.inplanes * 4,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            Norm(self.inplanes * 4),
        )

        self.layer3 = nn.Sequential(
            BasicBlock(
                Cin=self.inplanes * 2,
                Cout=self.inplanes * 4,
                stride=1,
                downsample=self.downsample3,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 4,
                Cout=self.inplanes * 4,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
            BasicBlock(
                Cin=self.inplanes * 4,
                Cout=self.inplanes * 4,
                stride=1,
                downsample=None,
                pad=1,
                dilation=1,
            ),
        )

        self.downsample4 = nn.Sequential(
            nn.Conv2d(
                self.inplanes * 4,
                self.inplanes * 4,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            Norm(self.inplanes * 4),
        )

        self.layer4 = nn.Sequential(
            BasicBlock(
                Cin=self.inplanes * 4,
                Cout=self.inplanes * 4,
                stride=1,
                downsample=self.downsample4,
                pad=1,
                dilation=2,
            ),
            BasicBlock(
                Cin=self.inplanes * 4,
                Cout=self.inplanes * 4,
                stride=1,
                downsample=None,
                pad=1,
                dilation=2,
            ),
            BasicBlock(
                Cin=self.inplanes * 4,
                Cout=self.inplanes * 4,
                stride=1,
                downsample=None,
                pad=1,
                dilation=2,
            ),
        )

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            convbn(128, 32, 1, 1, 0, 1),
            Act(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            convbn(128, 32, 1, 1, 0, 1),
            Act(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            Act(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1),
            Act(inplace=True),
        )

        self.lastconv = nn.Sequential(
            convbn(320, 128, 3, 1, 1, 1),
            Act(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
        )

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(
            output_branch1,
            (output_skip.size()[2], output_skip.size()[3]),
            mode="bilinear",
        )

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(
            output_branch2,
            (output_skip.size()[2], output_skip.size()[3]),
            mode="bilinear",
        )

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(
            output_branch3,
            (output_skip.size()[2], output_skip.size()[3]),
            mode="bilinear",
        )

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(
            output_branch4,
            (output_skip.size()[2], output_skip.size()[3]),
            mode="bilinear",
        )

        output_feature = torch.cat(
            (
                output_raw,
                output_skip,
                output_branch4,
                output_branch3,
                output_branch2,
                output_branch1,
            ),
            1,
        )
        output_feature = self.lastconv(output_feature)

        return output_feature
