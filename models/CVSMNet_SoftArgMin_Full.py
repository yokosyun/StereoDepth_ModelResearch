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


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out


class CVSMNet_SoftArgMin_Full(nn.Module):
    def __init__(self, maxdisp):
        super(CVSMNet_SoftArgMin_Full, self).__init__()
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

        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=2),
            ResBlock(32, 32, dilation=4),
            ResBlock(32, 32, dilation=8),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, padding=1),
            # nn.ReLU(),
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

    def correction(self, disp, img):

        disp_img = torch.cat([disp, img], axis=1)

        disp_full = self.refine(disp_img)

        return disp_full

    def forward(self, left, right):
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)

        cost_volume = self.create_costvolume(left_feature, right_feature)

        up1 = self.estimate_disparity(cost_volume)

        pred_left = self.disparity_regression(up1, left.size()[2], left.size()[3])

        pred_left = self.correction(pred_left, left)

        return pred_left