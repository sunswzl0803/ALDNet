import torch
from torch import nn


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
    )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()

        if ch_in == ch_out:
            self.up = nn.Upsample(scale_factor=2)  # 以最后一层为例子 1*1024*32*32 ->Upsample-> 1*1024*64*64
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
        )

    def forward(self, x, x_e):
        x1 = self.up(x)
        x2 = torch.add(x1, x_e)
        return [x1, x2]

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
#
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out
class PDFNet(nn.Module):  # 添加了空间注意力和通道注意力
    def __init__(self, img_ch=3, num_classes = 2, channels=[64, 128, 256, 512]):
        super(PDFNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)  # 64
        self.Conv2 = conv_block(ch_in=64, ch_out=128)  # 64 128
        self.Conv3 = conv_block(ch_in=128, ch_out=256)  # 128 256
        self.Conv4 = conv_block(ch_in=256, ch_out=512)  # 256 512
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)  # 512 1024
        self.ta1 = TripletAttention()
        self.ta2 = TripletAttention()
        self.ta3 = TripletAttention()
        self.ta4 = TripletAttention()

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.AFF4 = AFF(channels=channels[3], r=4)
        self.AFF3 = AFF(channels=channels[2], r=4)
        self.AFF2 = AFF(channels=channels[1], r=4)
        self.AFF1 = AFF(channels=channels[0], r=4)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)  # 2048 1024

        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)  # 2048 512

        self.Up4 = up_conv(ch_in=512, ch_out=256)  # 512 512
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)  # 256 128
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)  # 128 64
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # self.Up_conv1 = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )
        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # encoding path
        feat1 = self.Conv1(x)
        feat1 = self.ta1(feat1) + feat1

        x2 = self.Maxpool(feat1)
        feat2 = self.Conv2(x2)
        feat2 = self.ta1(feat2) + feat2

        x3 = self.Maxpool(feat2)
        feat3 = self.Conv3(x3)
        feat3 = self.ta1(feat3) + feat3

        x4 = self.Maxpool(feat3)
        feat4 = self.Conv4(x4)
        feat4 = self.ta1(feat4) + feat4

        x5 = self.Maxpool(feat4)
        feat5 = self.Conv5(x5)  #16 16 1024


        [d41, d42] = self.Up5(feat5, feat4) #d41:32 32 512  d42:32 32 152
        aff4 = self.AFF4(feat4, d41)
        x4 = torch.cat([d42, aff4], dim=1)
        x4c = self.Up_conv5(x4)  # 512

        [d31, d32] = self.Up4(x4c, feat3)
        du4 = self.up3(x4)
        aff3 = self.AFF3(feat3, du4)
        x3 = torch.cat([d32, aff3], dim=1)
        x3c = self.Up_conv4(x3)

        [d21, d22] = self.Up3(x3c, feat2)
        du3 = self.up2(x3)
        aff2 = self.AFF2(feat2, du3)
        x2 = torch.cat([d22, aff2], dim=1)
        x2c = self.Up_conv3(x2)

        [d11, d12] = self.Up2(x2c, feat1)
        du2 = self.up1(x2)
        aff1 = self.AFF1(feat1, du2)
        x1 = torch.cat([d12, aff1], dim=1)
        x1 = self.Up_conv2(x1)
        # x1 = self.Up_conv1(x1)
        x = self.Conv_1x1(x1)
        return x
def PDFnet(pretrained=False, **kwargs):
    model = PDFNet(img_ch=3, num_classes = 7, channels=[64, 128, 256,512])
    if pretrained:
        weights_path = '...'  # Replace this with the actual local file path
        model.load_state_dict(torch.load(weights_path))
    return model
