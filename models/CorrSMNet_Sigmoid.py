from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
import sys

sys.path.append("../")
from utils.activations_autofn import MishAuto
from utils.selectedNorm import *

Act = nn.ReLU
# Act = SwishAuto
# Act = MishAuto


class Corr1d(nn.Module):
    def __init__(self, kernel_size=1, stride=1, D=1, simfun=None):
        super(Corr1d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.D = D
        if simfun is None:
            self.simfun = self.simfun_default
        else:  # such as simfun = nn.CosineSimilarity(dim=1)
            self.simfun = simfun

    def simfun_default(self, fL, fR):
        return (fL * fR).sum(dim=1)

    def forward(self, fL, fR):
        bn, c, h, w = fL.shape
        D = self.D
        stride = self.stride
        kernel_size = self.kernel_size
        corrmap = fL.new_zeros([bn, D, h, w], requires_grad=False)
        corrmap[:, 0] = self.simfun(fL, fR)
        for i in range(1, D):
            if i >= w:
                break
            idx = i * stride
            corrmap[:, i, :, idx:] = self.simfun(fL[:, :, :, idx:], fR[:, :, :, :-idx])
        if kernel_size > 1:
            assert kernel_size % 2 == 1
            m = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
            corrmap = m(corrmap)
        return corrmap


class CorrSMNet_Sigmoid(nn.Module):
    def __init__(self, maxdisp):
        super(CorrSMNet_Sigmoid, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

        self.corr = Corr1d(kernel_size=1, stride=1, D=self.maxdisp // 4, simfun=None)

        self.maxpool = nn.MaxPool2d(2)

        in_channels = self.maxdisp // 4

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 4),
            nn.ReLU(inplace=True),
        )

        dilation = 1
        pad = 1

        self.bottom_11 = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=dilation if dilation > 1 else pad,
                dilation=dilation,
                bias=False,
            ),
            SelectedNorm(in_channels),
            nn.ReLU(inplace=True),
        )

        dilation = 3

        self.bottom_12 = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=dilation if dilation > 1 else pad,
                dilation=dilation,
                bias=False,
            ),
            SelectedNorm(in_channels),
            nn.ReLU(inplace=True),
        )

        dilation = 5

        self.bottom_13 = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=dilation if dilation > 1 else pad,
                dilation=dilation,
                bias=False,
            ),
            SelectedNorm(in_channels),
            nn.ReLU(inplace=True),
        )

        dilation = 7

        self.bottom_14 = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=dilation if dilation > 1 else pad,
                dilation=dilation,
                bias=False,
            ),
            SelectedNorm(in_channels),
            nn.ReLU(inplace=True),
        )

        self.bottom_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 8, in_channels * 4, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, padding=0),
            SelectedNorm(in_channels * 4),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels * 8, in_channels * 4, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, padding=1),
            SelectedNorm(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            SelectedNorm(in_channels),
            nn.ReLU(inplace=True),
        )

        self.classify = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
        )

    def estimate_disparity(self, corr):

        down1 = self.down1(corr)
        down2 = self.maxpool(down1)
        down2 = self.down2(down2)
        bottom_1 = self.maxpool(down2)

        bottom_11 = self.bottom_11(bottom_1)
        bottom_12 = self.bottom_12(bottom_1)
        bottom_13 = self.bottom_13(bottom_1)
        bottom_14 = self.bottom_14(bottom_1)

        bottom_out = torch.cat(
            [bottom_1, bottom_11, bottom_12, bottom_13, bottom_14], axis=1
        )
        bottom_out = self.bottom_fuse(bottom_out)

        up2 = F.interpolate(
            bottom_out, size=None, scale_factor=2, mode="bilinear", align_corners=None
        )
        up2 = torch.cat([up2, down2], axis=1)
        up2 = self.up2(up2)

        up1 = F.interpolate(
            up2, size=None, scale_factor=2, mode="bilinear", align_corners=None
        )
        up1 = torch.cat([up1, down1], axis=1)
        up1 = self.up1(up1)

        return up1

    def disparity_regression(self, input, height, width):

        left_disp = self.classify(input)
        left_disp = torch.sigmoid(left_disp)
        left_disp = left_disp * self.maxdisp
        if left_disp.ndim == 3:
            left_disp = torch.unsqueeze(left_disp, 0)
        left_disp = F.upsample(left_disp, [height, width], mode="bilinear")
        return left_disp

    def forward(self, left, right):

        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)

        corr = self.corr(left_feature, right_feature)

        up1 = self.estimate_disparity(corr)

        pred_left = self.disparity_regression(up1, left.size()[2], left.size()[3])

        return pred_left