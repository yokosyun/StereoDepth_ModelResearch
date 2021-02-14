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
import time

Act = nn.ReLU
# Act = SwishAuto
# Act = MishAuto


class CVSMNet_SoftArgMin_3DUNetSpace(nn.Module):
    def __init__(self, maxdisp):
        super(CVSMNet_SoftArgMin_3DUNetSpace, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.maxPool3d = nn.MaxPool3d(3, stride=(1, 2, 2), padding=(1, 1, 1))
        self.upsample3d = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")

        in_channels = 64

        self.down0 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
        )

        self.up1 = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
        )

        self.up0 = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
        )

        self.bottom1 = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
        )

        self.bottom2 = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
        )

        self.bottom3 = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
        )

        self.classify = nn.Sequential(
            nn.Conv3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.InstanceNorm3d(in_channels // 2),
            Act(inplace=True),
            nn.Conv3d(
                in_channels // 2, 1, kernel_size=3, padding=1, stride=1, bias=False
            ),
        )

        self.disparityregression = disparityregression(self.maxdisp // 4)

    def create_costvolume(self, refimg_fea, targetimg_fea):
        D = self.maxdisp // 4
        bn, c, h, w = refimg_fea.shape
        cost = refimg_fea.new_zeros([bn, 2 * c, D, h, w], requires_grad=False)

        for i in range(D):
            if i > 0:
                cost[:, : refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1] :, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, : refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1] :, i, :, :] = targetimg_fea

        cost = cost.contiguous()
        return cost

    def estimate_disparity(self, cost):
        # down0
        down0 = self.down0(cost)
        # print("down0.shape=",down0.shape)

        # down1
        down1 = self.maxPool3d(down0)
        # print("down1.shape=",down1.shape)
        down1 = self.down1(down1)
        # print("down1.shape=",down1.shape)

        # bottom
        bottom = self.maxPool3d(down1)
        # print("bottom.shape=",bottom.shape)
        bottom = self.bottom1(bottom)
        # print("bottom.shape=",bottom.shape)
        bottom = self.bottom2(bottom)
        # print("bottom.shape=",bottom.shape)
        bottom = self.bottom3(bottom)
        # print("bottom.shape=",bottom.shape)

        # up1
        up1 = self.upsample3d(bottom) + down1
        # print("up1.shape=",up1.shape)
        up1 = self.up1(up1)
        # print("up1.shape=",up1.shape)

        # up0
        up0 = self.upsample3d(up1) + down0
        # print("up0.shape=",up0.shape)
        up0 = self.up0(up0)
        # print("up0.shape=",up0.shape)

        return up0

    def disparity_regression(self, input, height, width):

        cost = self.classify(input)

        prob = F.softmax(-cost, 2)

        prob = torch.squeeze(prob, 0)

        left_disp = self.disparityregression(prob)

        if left_disp.ndim == 3:
            left_disp = torch.unsqueeze(left_disp, 0)
        left_disp = F.upsample(left_disp, [height, width], mode="bilinear")

        return left_disp * 4

    def forward(self, left, right):
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)

        cost_volume = self.create_costvolume(left_feature, right_feature)

        up1 = self.estimate_disparity(cost_volume)

        pred_left = self.disparity_regression(up1, left.size()[2], left.size()[3])

        return pred_left