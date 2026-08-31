import os.path
import warnings

import torch
from torch import nn
from networks.pvtv2 import *
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


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


# class Encoder(nn.Module):
#     def __init__(self, pretrain_path, in_chans):
#         super().__init__()
#         # self.backbone = pvt_v2_b2(in_chans=in_chans)
#         self.backbone = pvt_v2_b0(in_chans=in_chans)

#         if pretrain_path is None:
#             warnings.warn('please provide the pretrained pvt model. Not using pretrained model.')
#         elif not os.path.isfile(pretrain_path):
#             warnings.warn(f'path: {pretrain_path} does not exists. Not using pretrained model.')
#         else:
#             print(f"using pretrained file: {pretrain_path}")
#             save_model = torch.load(pretrain_path)
#             model_dict = self.backbone.state_dict()
#             state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
#             model_dict.update(state_dict)

#             self.backbone.load_state_dict(model_dict)

    # def forward(self, x):
    #     f1, f2, f3, f4 = self.backbone(x)  # (x: 3, 352, 352)
    #     return f1, f2, f3, f4

class Encoder(nn.Module):
    def __init__(self, pretrain_path, in_chans, backbone="b0"):
        super().__init__()
        backbone_map = {
            "b0": (pvt_v2_b0, [32, 64, 160, 256]),
            "b1": (pvt_v2_b1, [64, 128, 320, 512]),
            "b2": (pvt_v2_b2, [64, 128, 320, 512]),
            "b3": (pvt_v2_b3, [64, 128, 320, 512]),
            "b4": (pvt_v2_b4, [64, 128, 320, 512]),
            "b5": (pvt_v2_b5, [64, 128, 320, 512]),
        }
        backbone_fn, self.embed_dims = backbone_map[backbone]
        self.backbone = backbone_fn(in_chans=in_chans)
        # load weights (same as before...)

    def forward(self, x):
        outs = self.backbone(x)
        assert isinstance(outs, (list, tuple)), \
            f"Expected list of features, got {type(outs)}"
        assert len(outs) == 4, \
            f"Expected 4 feature maps, got {len(outs)}"
        return outs[0], outs[1], outs[2], outs[3]

class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class UNetV2(nn.Module):
    def __init__(self, in_chans=1, channel=32, n_classes=1,
                 deep_supervision=True, pretrained_path=None, backbone="b0"):
        super().__init__()
        self.deep_supervision = deep_supervision

        # build encoder
        self.encoder = Encoder(pretrained_path, in_chans, backbone)
        dims = self.encoder.embed_dims  # <-- get dims dynamically

        # attentions
        self.ca_1, self.sa_1 = ChannelAttention(dims[0]), SpatialAttention()
        self.ca_2, self.sa_2 = ChannelAttention(dims[1]), SpatialAttention()
        self.ca_3, self.sa_3 = ChannelAttention(dims[2]), SpatialAttention()
        self.ca_4, self.sa_4 = ChannelAttention(dims[3]), SpatialAttention()

        # translayers
        self.Translayer_1 = BasicConv2d(dims[0], channel, 1)
        self.Translayer_2 = BasicConv2d(dims[1], channel, 1)
        self.Translayer_3 = BasicConv2d(dims[2], channel, 1)
        self.Translayer_4 = BasicConv2d(dims[3], channel, 1)

        # SDI modules
        self.sdi_1, self.sdi_2 = SDI(channel), SDI(channel)
        self.sdi_3, self.sdi_4 = SDI(channel), SDI(channel)

        # segmentation outputs
        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, n_classes, 1, 1) for _ in range(4)
        ])

        # decoder upsampling
        self.deconv2 = nn.ConvTranspose2d(channel, channel, 4, 2, 1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, 4, 2, 1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, 4, 2, 1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, 4, 2, 1, bias=False)

    def forward(self, x):
        seg_outs = []
        f1, f2, f3, f4 = self.encoder(x)

        f1 = self.Translayer_1(self.sa_1(self.ca_1(f1) * f1) * f1)
        f2 = self.Translayer_2(self.sa_2(self.ca_2(f2) * f2) * f2)
        f3 = self.Translayer_3(self.sa_3(self.ca_3(f3) * f3) * f3)
        f4 = self.Translayer_4(self.sa_4(self.ca_4(f4) * f4) * f4)

        f41, f31, f21, f11 = (
            self.sdi_4([f1, f2, f3, f4], f4),
            self.sdi_3([f1, f2, f3, f4], f3),
            self.sdi_2([f1, f2, f3, f4], f2),
            self.sdi_1([f1, f2, f3, f4], f1),
        )

        seg_outs.append(self.seg_outs[0](f41))

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode="bilinear")

        return seg_outs[::-1] if self.deep_supervision else seg_outs[-1]



# if __name__ == "__main__":
#     pretrained_path = "/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/pvt_pth/pvt_v2_b2.pth"
#     model = UNetV2(n_classes=2, deep_supervision=True, pretrained_path=None)
#     x = torch.rand((2, 3, 256, 256))
#     ys = model(x)
#     for y in ys:
#         print(y.shape)
