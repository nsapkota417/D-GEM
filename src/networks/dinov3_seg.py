import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ViTTokenPyramid(nn.Module):
    def __init__(self, in_ch, fpn_ch=256):
        super().__init__()

        self.p16 = nn.Sequential(
            nn.Conv2d(in_ch, fpn_ch, 1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.GELU(),
        )

        self.p32 = nn.Sequential(
            ConvBNAct(fpn_ch, fpn_ch, k=3, s=2, p=1),
            ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1),
        )

        self.p8 = nn.Sequential(
            ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1),
        )

        self.p4 = nn.Sequential(
            ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1),
        )

    def forward(self, x16):
        p16 = self.p16(x16)
        p32 = self.p32(p16)

        p8 = F.interpolate(
            p16,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        p8 = self.p8(p8)

        p4 = F.interpolate(
            p16,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        p4 = self.p4(p4)

        return {
            "p32": p32,
            "p16": p16,
            "p8": p8,
            "p4": p4,
        }


class FPNDecoder(nn.Module):
    def __init__(self, fpn_ch=256, num_classes=7, head_ch=128):
        super().__init__()

        self.lat32 = nn.Conv2d(fpn_ch, fpn_ch, 1)
        self.lat16 = nn.Conv2d(fpn_ch, fpn_ch, 1)
        self.lat8 = nn.Conv2d(fpn_ch, fpn_ch, 1)
        self.lat4 = nn.Conv2d(fpn_ch, fpn_ch, 1)

        self.out32 = ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1)
        self.out16 = ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1)
        self.out8 = ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1)
        self.out4 = ConvBNAct(fpn_ch, fpn_ch, k=3, s=1, p=1)

        self.head = nn.Sequential(
            ConvBNAct(fpn_ch, head_ch, k=3, s=1, p=1),
            nn.Conv2d(head_ch, num_classes, kernel_size=1),
        )

    def forward(self, feats, out_hw):
        p32 = self.lat32(feats["p32"])
        p16 = self.lat16(feats["p16"])
        p8 = self.lat8(feats["p8"])
        p4 = self.lat4(feats["p4"])

        p16 = p16 + F.interpolate(
            p32,
            size=p16.shape[-2:],
            mode="nearest",
        )
        p8 = p8 + F.interpolate(
            p16,
            size=p8.shape[-2:],
            mode="nearest",
        )
        p4 = p4 + F.interpolate(
            p8,
            size=p4.shape[-2:],
            mode="nearest",
        )

        p32 = self.out32(p32)
        p16 = self.out16(p16)
        p8 = self.out8(p8)
        p4 = self.out4(p4)

        logits4 = self.head(p4)
        logits = F.interpolate(
            logits4,
            size=out_hw,
            mode="bilinear",
            align_corners=False,
        )

        return logits


class DINOv3ViTSeg(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        num_classes=7,
        patch_size=16,
        fpn_ch=256,
        pt_encoder=True,
        ft_encoder=False,
        in_chans=3,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = int(in_chans)

        if pt_encoder:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                token=True,
            )
        else:
            cfg = AutoConfig.from_pretrained(
                model_name,
                token=True,
            )
            self.encoder = AutoModel.from_config(cfg)

        # DINOv3 expects 3-channel RGB input.
        # If input is 4ch, learn a small 1x1 adapter:
        # 4ch -> 3ch -> DINOv3.
        if self.in_chans == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Conv2d(
                self.in_chans,
                3,
                kernel_size=1,
                bias=False,
            )

        self.num_register_tokens = getattr(
            self.encoder.config,
            "num_register_tokens",
            0,
        )
        embed_dim = self.encoder.config.hidden_size

        self.neck = ViTTokenPyramid(
            in_ch=embed_dim,
            fpn_ch=fpn_ch,
        )

        self.decoder = FPNDecoder(
            fpn_ch=fpn_ch,
            num_classes=num_classes,
        )

        if not ft_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        orig_H, orig_W = pixel_values.shape[-2:]

        # Convert non-3ch input to 3ch for pretrained DINOv3.
        pixel_values = self.input_adapter(pixel_values)

        B, _, H, W = pixel_values.shape

        out = self.encoder(pixel_values=pixel_values)
        tokens = out.last_hidden_state

        start = 1 + self.num_register_tokens
        patch_tokens = tokens[:, start:, :]

        gh, gw = H // self.patch_size, W // self.patch_size
        N = patch_tokens.shape[1]

        if N != gh * gw:
            raise ValueError(
                f"Token mismatch: N={N}, grid={gh}x{gw}={gh * gw}. "
                f"Likely due to non-multiple-of-patch input size."
            )

        x16 = patch_tokens.transpose(1, 2).reshape(
            B,
            -1,
            gh,
            gw,
        )

        feats = self.neck(x16)
        logits = self.decoder(feats, (orig_H, orig_W))

        return logits

    def encode_patch_tokens(self, pixel_values, pre_norm: bool = False):
        pixel_values = self.input_adapter(pixel_values)

        out = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=pre_norm,
        )

        if (
            pre_norm
            and hasattr(out, "hidden_states")
            and out.hidden_states is not None
        ):
            tokens = out.hidden_states[-1]
        else:
            tokens = out.last_hidden_state

        start = 1 + self.num_register_tokens
        return tokens[:, start:, :]

    def decode_from_patch_tokens(self, patch_tokens, H, W):
        gh, gw = H // self.patch_size, W // self.patch_size

        if patch_tokens.shape[1] != gh * gw:
            raise ValueError(
                f"Token mismatch: N={patch_tokens.shape[1]} "
                f"vs grid={gh}x{gw}={gh * gw}"
            )

        x16 = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            -1,
            gh,
            gw,
        )

        feats = self.neck(x16)
        return self.decoder(feats, (H, W))