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


class CVSMNet_SoftArgMin(nn.Module):
    def __init__(self, maxdisp):
        super(CVSMNet_SoftArgMin, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

        in_channels = 64

        self.dres0 = nn.Sequential(
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

        self.dres1 = nn.Sequential(
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

        self.dres2 = nn.Sequential(
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

        self.dres3 = nn.Sequential(
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

        self.dres4 = nn.Sequential(
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

        ##Norm Version
        # self.classify = nn.Sequential(
        #     nn.Conv3d(in_channels//2, in_channels//2, kernel_size=3, padding=1, stride=1,bias=False),
        #     nn.InstanceNorm3d(in_channels//2),
        #     Act(inplace=True),
        #     nn.Conv3d(in_channels//2, 1, kernel_size=3, padding=1, stride=1,bias=False),
        #     nn.InstanceNorm3d(1),
        #     )

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
        cost0 = self.dres0(cost)  # this layer should hangle matching mainly
        cost0 = self.dres1(cost0) + cost0  # refinement
        cost0 = self.dres2(cost0) + cost0  # refinement
        cost0 = self.dres3(cost0) + cost0  # refinement
        cost0 = self.dres4(cost0) + cost0  # refinement
        return cost0

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