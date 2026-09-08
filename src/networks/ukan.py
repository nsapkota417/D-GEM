"""U-KAN image-segmentation baseline.

This is a self-contained PyTorch implementation of the U-shaped convolutional
and KAN-token architecture described in the U-KAN paper.  It accepts arbitrary
input sizes; decoder stages are resized to their skip tensors rather than
assuming that every spatial dimension is divisible by 32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineLinear(nn.Module):
    """KAN linear layer with a fixed uniform B-spline grid."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, order: int = 3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.order = order
        knots = torch.linspace(-1.0, 1.0, grid_size + 1)
        step = knots[1] - knots[0]
        knots = torch.cat((knots[0] - step * torch.arange(order, 0, -1), knots,
                           knots[-1] + step * torch.arange(1, order + 1)))
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())
        coefficients = grid_size + order
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, coefficients))
        self.spline_scale = nn.Parameter(torch.empty(out_features, in_features))
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.ones_(self.spline_scale)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for degree in range(1, self.order + 1):
            left = (x - grid[:, : -(degree + 1)]) / (grid[:, degree:-1] - grid[:, : -(degree + 1)])
            right = (grid[:, degree + 1 :] - x) / (grid[:, degree + 1 :] - grid[:, 1:-degree])
            basis = left * basis[:, :, :-1] + right * basis[:, :, 1:]
        return basis.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.activation(x), self.base_weight)
        spline = F.linear(
            self._basis(x).reshape(x.shape[0], -1),
            (self.spline_weight * self.spline_scale.unsqueeze(-1)).reshape(self.out_features, -1),
        )
        return base + spline


class DepthwiseBNReLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.layers(features).flatten(2).transpose(1, 2)


class KANTokenBlock(nn.Module):
    def __init__(self, channels: int, drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = BSplineLinear(channels, channels)
        self.fc2 = BSplineLinear(channels, channels)
        self.fc3 = BSplineLinear(channels, channels)
        self.dw1 = DepthwiseBNReLU(channels)
        self.dw2 = DepthwiseBNReLU(channels)
        self.dw3 = DepthwiseBNReLU(channels)
        self.drop = nn.Dropout(drop)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        residual = tokens
        tokens = self.fc1(self.norm(tokens).reshape(-1, tokens.shape[-1])).reshape_as(tokens)
        tokens = self.dw1(tokens, height, width)
        tokens = self.fc2(tokens.reshape(-1, tokens.shape[-1])).reshape_as(tokens)
        tokens = self.dw2(tokens, height, width)
        tokens = self.fc3(tokens.reshape(-1, tokens.shape[-1])).reshape_as(tokens)
        tokens = self.dw3(tokens, height, width)
        return residual + self.drop(tokens)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class UKAN(nn.Module):
    """U-KAN segmentation network returning logits at the input resolution."""

    def __init__(self, num_classes: int, in_channels: int = 3, embed_dims: tuple[int, int, int] = (256, 320, 512), drop_rate: float = 0.0):
        super().__init__()
        c1, c2, c3 = embed_dims
        self.encoder1 = ConvBlock(in_channels, c1 // 8)
        self.encoder2 = ConvBlock(c1 // 8, c1 // 4)
        self.encoder3 = ConvBlock(c1 // 4, c1)
        self.patch3 = PatchEmbed(c1, c2)
        self.patch4 = PatchEmbed(c2, c3)
        self.enc_block = KANTokenBlock(c2, drop_rate)
        self.bottleneck = KANTokenBlock(c3, drop_rate)
        self.dec_block = KANTokenBlock(c2, drop_rate)
        self.dec_token_block = KANTokenBlock(c1, drop_rate)
        self.norm3, self.norm4 = nn.LayerNorm(c2), nn.LayerNorm(c3)
        self.dnorm3, self.dnorm4 = nn.LayerNorm(c2), nn.LayerNorm(c1)
        self.decoder1 = DecoderBlock(c3, c2)
        self.decoder2 = DecoderBlock(c2, c1)
        self.decoder3 = DecoderBlock(c1, c1 // 4)
        self.decoder4 = DecoderBlock(c1 // 4, c1 // 8)
        self.decoder5 = DecoderBlock(c1 // 8, c1 // 8)
        self.head = nn.Conv2d(c1 // 8, num_classes, 1)

    @staticmethod
    def _map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.reshape(tokens.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _resize(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        t1 = F.max_pool2d(self.encoder1(x), 2)
        t2 = F.max_pool2d(self.encoder2(t1), 2)
        t3 = F.max_pool2d(self.encoder3(t2), 2)
        tokens, height, width = self.patch3(t3)
        t4 = self._map(self.norm3(self.enc_block(tokens, height, width)), height, width)
        tokens, height, width = self.patch4(t4)
        out = self._map(self.norm4(self.bottleneck(tokens, height, width)), height, width)
        out = self._resize(self.decoder1(out), t4) + t4
        height, width = out.shape[-2:]
        tokens = out.flatten(2).transpose(1, 2)
        out = self._map(self.dnorm3(self.dec_block(tokens, height, width)), height, width)
        out = self._resize(self.decoder2(out), t3) + t3
        height, width = out.shape[-2:]
        tokens = out.flatten(2).transpose(1, 2)
        out = self._map(self.dnorm4(self.dec_token_block(tokens, height, width)), height, width)
        out = self._resize(self.decoder3(out), t2) + t2
        out = self._resize(self.decoder4(out), t1) + t1
        out = F.interpolate(self.decoder5(out), size=output_size, mode="bilinear", align_corners=False)
        return self.head(out)
