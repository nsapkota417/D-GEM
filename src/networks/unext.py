"""Self-contained UNeXt image-segmentation baseline.

It follows the UNeXt pattern of convolutional stages, shifted-MLP token
stages, and a skip-connected decoder, but relies only on PyTorch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep) / keep


class DepthwiseConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.conv(features).flatten(2).transpose(1, 2)


class ShiftMLP(nn.Module):
    def __init__(self, channels: int, drop_rate: float = 0.0, shift_size: int = 5):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(channels, channels), nn.Linear(channels, channels)
        self.depthwise, self.activation = DepthwiseConv(channels), nn.GELU()
        self.dropout, self.shift_size = nn.Dropout(drop_rate), shift_size

    def _shift(self, tokens: torch.Tensor, height: int, width: int, dimension: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        pad = self.shift_size // 2
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        features = F.pad(features, (pad, pad, pad, pad))
        groups = torch.chunk(features, self.shift_size, dim=1)
        shifted = [torch.roll(group, offset, dims=dimension) for group, offset in zip(groups, range(-pad, pad + 1))]
        features = torch.cat(shifted, dim=1)[:, :, pad : pad + height, pad : pad + width]
        return features.flatten(2).transpose(1, 2)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        tokens = self._shift(tokens, height, width, 2)
        tokens = self.dropout(self.activation(self.depthwise(self.fc1(tokens), height, width)))
        return self.dropout(self.fc2(self._shift(tokens, height, width, 3)))


class ShiftBlock(nn.Module):
    def __init__(self, channels: int, drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm, self.mlp, self.drop_path = nn.LayerNorm(channels), ShiftMLP(channels, drop_rate), DropPath(drop_path_rate)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens + self.drop_path(self.mlp(self.norm(tokens), height, width))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection, self.norm = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1), nn.LayerNorm(out_channels)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        features = self.projection(features)
        height, width = features.shape[-2:]
        return self.norm(features.flatten(2).transpose(1, 2)), height, width


class UNext(nn.Module):
    """UNeXt baseline that accepts arbitrary spatial sizes and returns logits."""

    def __init__(self, num_classes: int, in_channels: int = 3,
                 embed_dims: tuple[int, int, int] = (128, 160, 256),
                 drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        c1, c2, c3 = embed_dims
        if c1 % 8:
            raise ValueError("UNeXt's first embedding width must be divisible by 8.")
        small, medium = c1 // 8, c1 // 4
        self.encoder1, self.encoder2, self.encoder3 = nn.Conv2d(in_channels, small, 3, padding=1), nn.Conv2d(small, medium, 3, padding=1), nn.Conv2d(medium, c1, 3, padding=1)
        self.encoder_norms = nn.ModuleList([nn.BatchNorm2d(small), nn.BatchNorm2d(medium), nn.BatchNorm2d(c1)])
        self.patch3, self.patch4 = PatchEmbed(c1, c2), PatchEmbed(c2, c3)
        self.enc_block, self.bottleneck = ShiftBlock(c2, drop_rate, drop_path_rate / 2), ShiftBlock(c3, drop_rate, drop_path_rate)
        self.dec_block1, self.dec_block2 = ShiftBlock(c2, drop_rate, drop_path_rate / 2), ShiftBlock(c1, drop_rate)
        self.norm3, self.norm4, self.dec_norm1, self.dec_norm2 = nn.LayerNorm(c2), nn.LayerNorm(c3), nn.LayerNorm(c2), nn.LayerNorm(c1)
        self.decoder1, self.decoder2 = nn.Conv2d(c3, c2, 3, padding=1), nn.Conv2d(c2, c1, 3, padding=1)
        self.decoder3, self.decoder4, self.decoder5 = nn.Conv2d(c1, medium, 3, padding=1), nn.Conv2d(medium, small, 3, padding=1), nn.Conv2d(small, small, 3, padding=1)
        self.head = nn.Conv2d(small, num_classes, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            if module.bias is not None: nn.init.zeros_(module.bias)

    @staticmethod
    def _map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.reshape(tokens.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _up(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(features, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        t1 = F.relu(F.max_pool2d(self.encoder_norms[0](self.encoder1(x)), 2))
        t2 = F.relu(F.max_pool2d(self.encoder_norms[1](self.encoder2(t1)), 2))
        t3 = F.relu(F.max_pool2d(self.encoder_norms[2](self.encoder3(t2)), 2))
        tokens, height, width = self.patch3(t3)
        t4 = self._map(self.norm3(self.enc_block(tokens, height, width)), height, width)
        tokens, height, width = self.patch4(t4)
        out = self._map(self.norm4(self.bottleneck(tokens, height, width)), height, width)
        out = self._up(F.relu(self.decoder1(out)), t4) + t4
        height, width = out.shape[-2:]
        out = self._map(self.dec_norm1(self.dec_block1(out.flatten(2).transpose(1, 2), height, width)), height, width)
        out = self._up(F.relu(self.decoder2(out)), t3) + t3
        height, width = out.shape[-2:]
        out = self._map(self.dec_norm2(self.dec_block2(out.flatten(2).transpose(1, 2), height, width)), height, width)
        out = self._up(F.relu(self.decoder3(out)), t2) + t2
        out = self._up(F.relu(self.decoder4(out)), t1) + t1
        return self.head(F.interpolate(F.relu(self.decoder5(out)), size=output_size, mode="bilinear", align_corners=False))
