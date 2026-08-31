# Model: U-KAST (U-shaped Kolmogorov–Arnold Swin Transformer)
#
# Based on:
#   - "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
#       Hatamizadeh et al., https://arxiv.org/abs/2201.01266
#
#   - "Kolmogorov–Arnold Transformer"
#       Xingyi Yang, Xinchao Wang, https://arxiv.org/pdf/2409.10594
#
# Original implementation: MONAI SwinUNETR
# Modified by: Nishchal Sapkota (nsapkota@nd.edu)

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from functools import partial
from timm.models.layers import to_2tuple
from networks.rational_kat_cu.kat_rational import KAT_Group as KAT_Group_Triton
from networks.rational_kat_cu.kat_rational import KAT_Group_Torch as KAT_Group_Torch


rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "UKAST",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinKATBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]

class RegisterNorm(nn.Module):
    def __init__(self, num_features, reg_dim, hidden_dim=128, spatial_dims=3):
        super().__init__()
        if spatial_dims == 3:
            self.norm = nn.InstanceNorm3d(num_features, affine=False)
            extra_dims = (1, 1, 1)
        else:
            self.norm = nn.InstanceNorm2d(num_features, affine=False)
            extra_dims = (1, 1)
        self.extra_dims = extra_dims
        self.mlp_gamma = nn.Sequential(
            nn.Linear(reg_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_features)
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(reg_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x, reg_tokens):
        reg = reg_tokens.mean(dim=1)
        shape = (reg.size(0), -1) + self.extra_dims
        gamma = self.mlp_gamma(reg).view(*shape)
        beta  = self.mlp_beta(reg).view(*shape)
        return self.norm(x) * (1 + gamma) + beta

class DecoderWithRegisterNorm(nn.Module):
    def __init__(self, base_decoder, reg_dim, spatial_dims=3):
        super().__init__()
        self.base = base_decoder
        self.norm = RegisterNorm(base_decoder.out_channels, reg_dim, spatial_dims=spatial_dims)
    def forward(self, x, skip, reg_tokens):
        x = self.base(x, skip)
        if reg_tokens is not None:
            x = self.norm(x, reg_tokens)
        return x

class KAN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            act_init="gelu",
            device=None,
            num_groups=8,
            poly_order=(5,4),
            use_triton=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        KAT_Group = KAT_Group_Triton if use_triton else KAT_Group_Torch

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = KAT_Group(
            mode="identity", 
            device=device,
            num_groups=num_groups,
            poly_order=poly_order
        )
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act2 = KAT_Group(
            mode=act_init, 
            device=device,
            num_groups=num_groups,
            poly_order=poly_order
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

class UKASTdb(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device,
        patch_size: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str | nn.Module = "merging",
        use_resconv: bool = False,
        use_postnorm: bool = False,

        # register specific
        num_registers: int = 0,
        register_stages: set[str] = {"bn"},

        # kan specific
        use_triton: bool = True,
        use_kan: bool=False,
        kan_act_init: str = 'swish',
        kan_group_size: int = 8,
        kan_poly_order: tuple | int = (5,4),

        # spade specific
        use_register_norm: bool=False, 

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            patch_size: size of the patch token.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            norm_layer: normalization layer.
            patch_norm: whether to apply normalization to the patch embedding. Default is False.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_resconv: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = UKAST(in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = UKAST(in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = UKAST(in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size

        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinKAT = SwinKAT(
            device=device,
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_resconv=use_resconv,
            use_postnorm=use_postnorm,
            use_kan=use_kan,
            use_triton=use_triton,
            num_registers=num_registers,
            kan_act_init=kan_act_init,
            kan_group_size=kan_group_size,
            kan_poly_order=kan_poly_order        
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        if num_registers > 0:
            in_dim  = feature_size * 2**(len(depths) - 1)   # deepest stage reg dim
            out_dim = 16 * feature_size                     # bottleneck channels
            self.reg_bottleneck_proj = nn.Linear(in_dim, out_dim)
        else:
            self.reg_bottleneck_proj = None

        # if num_registers > 0:
        #     self.reg_fusion = RegisterFusion(
        #         dim=16 * feature_size,   # bottleneck channels
        #         reg_dim=feature_size * 2**(len(depths)-1),  # deepest reg dim
        #         num_heads=4
        #     )
        # else:
        #     self.reg_fusion = nn.Identity()
        self.register_stages = set(register_stages)
        if num_registers > 0:
            reg_dim = feature_size * 2**(len(depths) - 1)
            self.reg_fusions = nn.ModuleDict()

            if "bn" in self.register_stages:
                self.reg_fusions["bn"] = RegisterFusion(16*feature_size, reg_dim, num_heads=4)
            if "dec5" in self.register_stages:
                self.reg_fusions["dec5"] = RegisterFusion(8*feature_size, reg_dim, num_heads=4)
            if "dec4" in self.register_stages:
                self.reg_fusions["dec4"] = RegisterFusion(4*feature_size, reg_dim, num_heads=4)
            if "dec3" in self.register_stages:
                self.reg_fusions["dec3"] = RegisterFusion(2*feature_size, reg_dim, num_heads=4)
            if "dec2" in self.register_stages:
                self.reg_fusions["dec2"] = RegisterFusion(1*feature_size, reg_dim, num_heads=4)
            if "dec1" in self.register_stages:
                self.reg_fusions["dec1"] = RegisterFusion(1*feature_size, reg_dim, num_heads=4)
        else:
            self.reg_fusions = nn.ModuleDict()

        # spade specific
        # compute reg_dim regardless of num_registers
        reg_dim_all = feature_size * 2**(len(depths) - 1)

        self.use_register_norm = use_register_norm
        if use_register_norm:
            self.decoder5 = DecoderWithRegisterNorm(self.decoder5, reg_dim_all, spatial_dims=spatial_dims)
            self.decoder4 = DecoderWithRegisterNorm(self.decoder4, reg_dim_all, spatial_dims=spatial_dims)
            self.decoder3 = DecoderWithRegisterNorm(self.decoder3, reg_dim_all, spatial_dims=spatial_dims)
            self.decoder2 = DecoderWithRegisterNorm(self.decoder2, reg_dim_all, spatial_dims=spatial_dims)
            self.decoder1 = DecoderWithRegisterNorm(self.decoder1, reg_dim_all, spatial_dims=spatial_dims)


    def _decode(self, decoder, x, skip, reg_tokens):
        # unified call: works for both wrapped and plain decoders
        if isinstance(decoder, DecoderWithRegisterNorm):
            return decoder(x, skip, reg_tokens)
        return decoder(x, skip)

    def load_from(self, weights):
        raise NotImplementedError("Pretrained SwinUNETR weight loading is not supported for UKAST.")

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])

        # ---- Swin-KAT backbone ----
        hidden_states_out, reg_tokens = self.swinKAT(x_in, self.normalize)

        # ---- Encoder pathway ----
        enc0 = self.encoder1(x_in)                  # shallow features
        enc1 = self.encoder2(hidden_states_out[0])  # stage 1 features
        enc2 = self.encoder3(hidden_states_out[1])  # stage 2 features
        enc3 = self.encoder4(hidden_states_out[2])  # stage 3 features

        # ---- Bottleneck ----
        dec4 = self.encoder10(hidden_states_out[4]) # deepest features

        # ---- Register fusion into bottleneck ----
        if reg_tokens is not None and "bn" in self.register_stages:
            dec4 = self.reg_fusions["bn"](dec4, reg_tokens)

        # ---- Decoder pathway ----
        # dec3 = self.decoder5(dec4, hidden_states_out[3]) # upsample + skip from stage 4
        # dec2 = self.decoder4(dec3, enc3)                 # skip from stage 3
        # dec1 = self.decoder3(dec2, enc2)                 # skip from stage 2
        # dec0 = self.decoder2(dec1, enc1)                 # skip from stage 1
        # out  = self.decoder1(dec0, enc0)                 # final upsample + shallow skip

        # ---- Decoder pathway ----
        dec3 = self._decode(self.decoder5, dec4, hidden_states_out[3], reg_tokens)
        if reg_tokens is not None and "dec5" in self.register_stages:
            dec3 = self.reg_fusions["dec5"](dec3, reg_tokens)

        dec2 = self._decode(self.decoder4, dec3, enc3, reg_tokens)
        if reg_tokens is not None and "dec4" in self.register_stages:
            dec2 = self.reg_fusions["dec4"](dec2, reg_tokens)

        dec1 = self._decode(self.decoder3, dec2, enc2, reg_tokens)
        if reg_tokens is not None and "dec3" in self.register_stages:
            dec1 = self.reg_fusions["dec3"](dec1, reg_tokens)

        dec0 = self._decode(self.decoder2, dec1, enc1, reg_tokens)
        if reg_tokens is not None and "dec2" in self.register_stages:
            dec0 = self.reg_fusions["dec2"](dec0, reg_tokens)

        out = self._decode(self.decoder1, dec0, enc0, reg_tokens)
        if reg_tokens is not None and "dec1" in self.register_stages:
            out = self.reg_fusions["dec1"](out, reg_tokens)

        # ---- Output head ----
        logits = self.out(out)
        return logits


class RegisterFusion(nn.Module):
    def __init__(self, dim, reg_dim, num_heads=4):
        super().__init__()
        self.reg_proj = nn.Linear(reg_dim, dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat, reg_tokens):
        B, C = feat.size(0), feat.size(1)
        spatial = feat.shape[2:]                  # works for 2D (H,W) or 3D (D,H,W)
        L = int(np.prod(spatial))
        x = feat.view(B, C, L).transpose(1, 2)    # [B, L, C]
        reg_proj = self.reg_proj(reg_tokens)      # [B, R, C]
        fused, _ = self.cross_attn(query=x, key=reg_proj, value=reg_proj)
        fused = self.norm(x + fused)
        fused = fused.transpose(1, 2).view(B, C, *spatial)
        return fused



def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()  # length 4 or 5 only
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    else:  # if len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)

    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore[operator]
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinKATBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        base_embed_dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        device,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        use_postnorm: bool = False,
        use_triton: bool = True,
        num_registers: int = 0,
        use_kan: bool = False,
        kan_act_init: str = 'swish',
        kan_group_size: int = 8,
        kan_poly_order: tuple | int = (5,4),
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.use_kan = use_kan
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # updated mlp with Group KAN (GRKAN)

        if use_kan:
            self.ffn = KAN(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                out_features=dim,
                drop=drop,
                device=device,
                act_init=kan_act_init,
                num_groups=kan_group_size,
                poly_order=kan_poly_order ,
                use_triton=use_triton
            )
        else:
            self.ffn = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")


        self.use_postnorm = use_postnorm

        # tiny cross-attn for regs
        # self.reg_in_proj = nn.Linear(base_embed_dim, dim, bias=False)
        if num_registers>0:
            self.reg_ln = nn.LayerNorm(dim)
            self.reg_q = nn.Linear(dim, dim)
            self.reg_k = nn.Linear(dim, dim)
            self.reg_v = nn.Linear(dim, dim)
            self.reg_proj = nn.Linear(dim, dim)
            self.reg_attn_drop = nn.Dropout(attn_drop)
            self.reg_proj_drop = nn.Dropout(drop)

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()

        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        else:  # elif len(x_shape) == 4
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    # NS EDITS: to account for KAN 
    def forward_part2(self, x):
        if self.use_kan:
            x_shape = x.shape

            if len(x_shape) == 4:  # 2D: [B, H, W, C]
                B, H, W, C = x_shape
                x = x.view(B, H * W, C)     # -> [B, L, C], L = H*W
                x = self.ffn(x)             # apply KAN
                x = x.view(B, H, W, C)      # back to [B, H, W, C]

            elif len(x_shape) == 5:  # 3D: [B, D, H, W, C]
                B, D, H, W, C = x_shape
                x = x.view(B, D * H * W, C) # -> [B, L, C], L = D*H*W
                x = self.ffn(x)
                x = x.view(B, D, H, W, C)   # back to [B, D, H, W, C]

            else:
                raise ValueError(f"Unexpected x shape {x_shape} in forward_part2")
        else:
            x = self.ffn(x)

        return self.drop_path(x)

    def forward(self, x, mask_matrix, reg_tokens=None):
        # import IPython; IPython.embed()
        # ---- Spatial branch (unaltered Swin) ----
        # ---- Attention branch ----
        if not self.use_postnorm:   # pre-norm
            x = self.norm1(x)

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_postnorm:   # post-norm
            x = self.norm1(x)

        # ---- MLP branch on spatial tokens ----
        if not self.use_postnorm:
            x = self.norm2(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        if self.use_postnorm:
            x = self.norm2(x)

        # ---- Global register update (NO windows) ----
        if reg_tokens is not None:
            B = x.size(0); C = x.size(-1)
            if x.dim() == 5:
                _, D, H, W, _ = x.shape
                x_flat = x.view(B, D * H * W, C)  # [B, N, C]
            else:
                _, H, W, _ = x.shape
                x_flat = x.view(B, H * W, C)      # [B, N, C]

            # if reg_tokens.size(-1) != C:
            #     # only project if input != C
            #     in_dim = reg_tokens.size(-1)
            #     if in_dim != C:
            #         reg_tokens = nn.functional.linear(
            #             reg_tokens, 
            #             self.reg_in_proj.weight[:, :in_dim]  # slice to match current in_dim
            #         )

            # LN on regs (optional but stabilizing)
            reg = self.reg_ln(reg_tokens)

            # single-head dot-product cross-attn: regs (queries) over spatial (keys/vals)
            q = self.reg_q(reg)          # [B, R, C]
            k = self.reg_k(x_flat)       # [B, N, C]
            v = self.reg_v(x_flat)       # [B, N, C]

            attn = torch.softmax((q @ k.transpose(-2, -1)) * (C ** -0.5), dim=-1)  # [B, R, N]
            attn = self.reg_attn_drop(attn)
            upd = self.reg_proj_drop(self.reg_proj(attn @ v))
            reg_tokens = reg_tokens + self.drop_path(upd)

        return x, reg_tokens



class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        device,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        use_postnorm: bool = False,
        use_kan: bool = False,
        num_registers: int=0,
        use_triton: bool = True,
        base_embed_dim: int | None = None,   # 👈 new
        reg_in_dim=None,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinKATBlock(
                    device=device,
                    dim=dim,
                    base_embed_dim=base_embed_dim,  # 👈 new
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    use_postnorm=use_postnorm,
                    use_kan=use_kan,
                    use_triton=use_triton,
                    num_registers=num_registers
                )
                for i in range(depth)
            ]
        )

        if num_registers > 0:
            if reg_in_dim is not None and reg_in_dim != dim:
                self.reg_proj = nn.Linear(reg_in_dim, dim, bias=False)
            else:
                self.reg_proj = nn.Identity()

        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))



    def forward(self, x, reg_tokens=None):
        x_shape = x.size()

        if reg_tokens is not None:
            if reg_tokens.size(-1) != self.blocks[0].dim:
                reg_tokens = self.reg_proj(reg_tokens)  # project only when mismatch

        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x, reg_tokens = blk(x, attn_mask, reg_tokens)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x, reg_tokens = blk(x, attn_mask, reg_tokens)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x, reg_tokens


class SwinKAT(nn.Module):
    
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        device,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_resconv: bool = False,
        use_postnorm: bool = False,
        num_registers: int = 0,
        use_kan: bool=False,
        use_triton: bool = True, 
        kan_act_init: str = 'swish',
        kan_group_size: int = 8,
        kan_poly_order: tuple | int = (5,4)      
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_resconv: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.use_resconv = use_resconv
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        if self.use_resconv:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        reg_dim = embed_dim
        for i_layer in range(self.num_layers):
            dim=int(embed_dim * 2**i_layer)
            layer = BasicLayer(
                device=device,
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
                use_postnorm=use_postnorm,
                use_kan=use_kan,
                use_triton=use_triton,
                base_embed_dim=embed_dim,   # 👈 pass here
                reg_in_dim=reg_dim,        # 👈 current reg dim
                num_registers=num_registers,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

            reg_dim = dim   # 👈 update for next stage

            if self.use_resconv:
                layerc = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Register tokens (learnable, shared across batch)
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, self.embed_dim)
            )
            trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            # Force trace() to generate a constant by casting to int
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):

        x0 = self.patch_embed(x)
        B, C = x0.shape[0], x0.shape[1]
        x0 = self.pos_drop(x0)

        # ---- Create registers ----
        reg_tokens = None
        if self.register_tokens is not None:
            reg_tokens = self.register_tokens.expand(B, -1, -1)   # [B, R, C]

        x0_out = self.proj_out(x0, normalize)

        # Stage 1
        if self.use_resconv:
            x0 = self.layers1c[0](x0.contiguous())
        x1, reg_tokens = self.layers1[0](x0.contiguous(), reg_tokens=reg_tokens)
        x1_out = self.proj_out(x1, normalize)

        # Stage 2
        if self.use_resconv:
            x1 = self.layers2c[0](x1.contiguous())
        x2, reg_tokens = self.layers2[0](x1.contiguous(), reg_tokens=reg_tokens)
        x2_out = self.proj_out(x2, normalize)

        # Stage 3
        if self.use_resconv:
            x2 = self.layers3c[0](x2.contiguous())
        x3, reg_tokens = self.layers3[0](x2.contiguous(), reg_tokens=reg_tokens)
        x3_out = self.proj_out(x3, normalize)


        # Stage 4
        if self.use_resconv:
            x3 = self.layers4c[0](x3.contiguous())

        x4, reg_tokens = self.layers4[0](x3.contiguous(), reg_tokens=reg_tokens)
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out], reg_tokens


def filter_ukast(key, value):
    """
    A filter function used to filter the pretrained weights from [1], then the weights can be loaded into MONAI SwinUNETR Model.
    This function is typically used with `monai.networks.copy_model_state`
    [1] "Valanarasu JM et al., Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training
    <https://arxiv.org/abs/2307.16896>"

    Args:
        key: the key in the source state dict used for the update.
        value: the value in the source state dict used for the update.

    Examples::

        import torch
        from monai.apps import download_url
        from monai.networks.utils import copy_model_state
        from monai.networks.nets.swin_unetr import SwinUNETR, filter_swinunetr

        model = ukast(in_channels=1, out_channels=3, feature_size=48)
        resource = (
            "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
        )
        ssl_weights_path = "./ssl_pretrained_weights.pth"
        download_url(resource, ssl_weights_path)
        ssl_weights = torch.load(ssl_weights_path, weights_only=True)["model"]

        dst_dict, loaded, not_loaded = copy_model_state(model, ssl_weights, filter_func=filter_swinunetr)

    """
    if key in [
        "encoder.mask_token",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "out.conv.conv.weight",
        "out.conv.conv.bias",
    ]:
        return None

    if key[:8] == "encoder.":
        if key[8:19] == "patch_embed":
            new_key = "swinViT." + key[8:]
        else:
            new_key = "swinViT." + key[8:18] + key[20:]

        return new_key, value
    else:
        return None