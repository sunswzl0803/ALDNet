from nets.res2net import res2net50_v1b_26w_4s
import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x
class AAF(nn.Module):
    def __init__(self, P):
        super(AAF, self).__init__()
        self.P = P
        self.n = P.size()[0]
        self.F = [nn.ELU(), nn.Hardshrink(), nn.Hardtanh(), nn.LeakyReLU(), nn.LogSigmoid(),
                  nn.ReLU(), nn.ReLU6(), nn.RReLU(), nn.SELU(), nn.CELU(), nn.Sigmoid(),
                  nn.Softplus(), nn.Softshrink(), nn.Softsign(), nn.Tanh(), nn.Tanhshrink()]

    def forward(self, x):
        sz = x.size()
        res = torch.zeros(sz).cuda()
        for i in range(self.n):
            res += self.P[i] * self.F[i](x)

        return res
class SRFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SRFE, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel*4, out_channel, 3, padding=1)

        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x4 = self.branch4(self.conv(x) + x3)
        x_cat = self.relu(self.gamma*self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1)) + (1-self.gamma)*x0)
        return x_cat
# ---------------------------------------旧DFM----------------
class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 2, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
class PBGnet(nn.Module):
    def __init__(self, num_classes=2, channel=32, pretrained=False, backbone='res2net'):
        super(PBGnet, self).__init__()
        if backbone == 'res2net':
            self.res2net = res2net50_v1b_26w_4s(pretrained=pretrained)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50...'.format(backbone))
        # self.up_conv = nn.Sequential(
        #
        #     nn.Conv2d(2048, 2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     # nn.UpsamplingBilinear2d(scale_factor=4),
        #     # nn.Conv2d(32, 2, kernel_size=3, padding=1),
        #     # nn.ReLU(),
        # )
        self.srfe2_1 = SRFE(512, channel)
        self.srfe3_1 = SRFE(1024, channel)
        self.srfe4_1 = SRFE(2048, channel)

        self.agg1 = aggregation(channel)
        self.P = nn.Parameter(torch.ones(15, 16).cuda() * (1 / 16))
        self.af = list()
        for i in range(15):
            self.af.append(AAF(self.P[i, :]))
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=3, padding=1)
        # self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, num_classes, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, num_classes, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, num_classes, kernel_size=3, padding=1)
        # ---- reverse attention branch 1 ----
        self.ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(64, num_classes, kernel_size=3, padding=1)
        # ---- reverse attention branch 0 ----
        self.ra0_conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.ra0_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra0_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra0_conv4 = BasicConv2d(64, num_classes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.backbone = backbone

    def forward(self, inputs):
        feats = []
        if self.backbone == 'res2net':
            feats = self.res2net.forward(inputs)
        feat1 = feats[0]
        feat2 = feats[1]
        feat3 = feats[2]
        feat4 = feats[3]
        feat5 = feats[4]
        x2_srfe = self.srfe2_1(feat3)  # channel -> 32              #512，32，32---- 32，32，32
        x3_srfe = self.srfe3_1(feat4)  # channel -> 32              #1024，16，16----32，16，16
        x4_srfe = self.srfe4_1(feat5)  # channel -> 32              #2048，8，8 -----32，8，8
        fg = self.agg1(x4_srfe, x3_srfe, x2_srfe)    #2，32，32
        # x6 = F.interpolate(fg, scale_factor=32, mode='bilinear', align_corners=False)
        crop_4 = F.interpolate(fg, scale_factor=0.25, mode='bilinear')  # 2，32，32---2，8，8
        x6 = F.interpolate(fg, scale_factor=8, mode='bilinear')  #2，256，256
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.repeat(1, 1024, 1, 1).mul(feat5)
        x = self.af[0](self.ra4_conv1(x))
        x = self.af[1](self.ra4_conv2(x))
        # x = F.relu(self.ra4_conv3(x))
        # x = F.relu(self.ra4_conv4(x))
        x43 = self.af[2](self.ra4_conv5(x))
        x = crop_4 + x43
        x5 = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)

        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.repeat(1, 512, 1, 1).mul(feat4)
        x = self.af[3](self.ra3_conv1(x))
        x = self.af[4](self.ra3_conv2(x))
        # x = F.relu(self.ra3_conv3(x))
        x33 = self.af[5](self.ra3_conv4(x))
        x = crop_3 + x33
        x4 = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)

        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.repeat(1, 256, 1, 1).mul(feat3)
        x = self.af[6](self.ra2_conv1(x))
        x = self.af[7](self.ra2_conv2(x))
        # x = F.relu(self.ra2_conv3(x))
        x23 = self.af[8](self.ra2_conv4(x))
        x = crop_2 + x23
        x3 = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

        crop_1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_1)) + 1
        x = x.repeat(1, 128, 1, 1).mul(feat2)
        x = self.af[9](self.ra1_conv1(x))
        x = self.af[10](self.ra1_conv2(x))
        # x = F.relu(self.ra1_conv3(x))
        x13 = self.af[11](self.ra1_conv4(x))
        x = crop_1 + x13
        x2 = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        crop_0 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_0)) + 1
        x = x.repeat(1, 32, 1, 1).mul(feat1)
        x = self.af[12](self.ra0_conv1(x))
        x = self.af[13](self.ra0_conv2(x))
        # x = F.relu(self.ra0_conv3(x))
        x03 = self.af[14](self.ra0_conv4(x))
        x = crop_0 + x03
        logits = F.interpolate(x, scale_factor=2, mode='bilinear')
        return logits, x2, x3, x4, x5, x6

