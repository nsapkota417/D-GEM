# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
import sys
import os

import torch.nn as nn
import torch

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

# Add KAT path for import
kat_path = os.path.join(os.path.dirname(__file__), 'kat')
if kat_path not in sys.path:
    sys.path.insert(0, kat_path)

# Import KAT components
from networks.kat import KATVisionTransformer


class UKAT(nn.Module):
    """
    U-KAT (U-Net with Kolmogorov-Arnold Transformer) based on UNETR architecture
    but replacing the ViT backbone with KAT (Kolmogorov-Arnold Transformer).
    
    This model combines:
    - KAT (Kolmogorov-Arnold Transformer) as the backbone encoder
    - UNETR's CNN-based decoders and skip connections
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 192,  # KAT tiny embed_dim
        mlp_ratio: float = 2.0,  # KAT mlp_ratio (mlp_dim = hidden_size * mlp_ratio)
        num_heads: int = 3,      # KAT tiny num_heads
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,   # Support 2D and 3D
        qkv_bias: bool = False,
        save_attn: bool = False,
        depth: int = 12,         # KAT depth (number of transformer layers)
        act_init: str = 'swish', # KAT activation initialization,
        num_registers: int = 1   # <-- handled inside KAT
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer (KAT embed_dim). Defaults to 192.
            mlp_ratio: ratio of mlp hidden dim to embedding dim. Defaults to 2.0.
            num_heads: number of attention heads. Defaults to 3.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Supports 2D and 3D. Defaults to 2.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
            depth: number of transformer layers. Defaults to 12.
            act_init: KAT activation initialization type. Defaults to 'swish'.
            num_registers: number of register tokens (like DINOv2/v3).
        """

        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("UKAT supports 2D and 3D spatial dimensions.")
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = depth
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.num_registers = num_registers
        self.classification = False

        # Initialize KAT backbone with MONAI support for 2D/3D
        # Note: reg_tokens handles register allocation internally
        self.kat = KATVisionTransformer(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=dropout_rate,
            proj_drop_rate=dropout_rate,
            attn_drop_rate=dropout_rate,
            drop_path_rate=0.0,
            act_init=act_init,
            weight_init="kan_mimetic",
            class_token=False,     # Disable class token for segmentation
            global_pool='',        # Use empty global_pool when class_token=False
            spatial_dims=spatial_dims,
            proj_type=proj_type,
            pos_embed_type="learnable",
            num_classes=0,         # No classification for segmentation
            reg_tokens=num_registers,  # <-- registers managed by KAT
        )

        # UNETR encoders (same as original)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        # UNETR decoders (same as original)
        self.decoder5 = UnetrUpBlock(spatial_dims, hidden_size, feature_size * 8, 3, 2, norm_name, res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims, feature_size * 8, feature_size * 4, 3, 2, norm_name, res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims, feature_size * 4, feature_size * 2, 3, 2, norm_name, res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims, feature_size * 2, feature_size, 3, 2, norm_name, res_block)
        self.out = UnetOutBlock(spatial_dims, feature_size, out_channels)

        # Projection settings (same as original UNETR)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        """Project features to match UNETR's expected format"""
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        """
        Forward pass through UKAT model.
        
        Args:
            x_in: input tensor
            
        Returns:
            output tensor after segmentation
        """
        # Get KAT features and intermediate layers
        # We need layers 3, 6, 9 for UNETR skip connections (0-indexed: 2, 5, 8)
        x, hidden_states_out = self.kat.forward_intermediates(
            x_in, 
            indices=[2, 5, 8, 11],
            return_prefix_tokens=False,
            norm=False,
            stop_early=False,
            output_fmt='NLC',
            intermediates_only=False
        )

        # Strip registers from transformer outputs before CNN encoders
        def strip_registers(h):
            return h[:, :-self.num_registers, :] if self.num_registers > 0 else h

        hidden_states_dict = {
            3: strip_registers(hidden_states_out[0]),
            6: strip_registers(hidden_states_out[1]),
            9: strip_registers(hidden_states_out[2]),
        }
        x = strip_registers(x)  # final features

        # UNETR forward pass with KAT features
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states_dict[3]))
        enc3 = self.encoder3(self.proj_feat(hidden_states_dict[6]))
        enc4 = self.encoder4(self.proj_feat(hidden_states_dict[9]))

        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        return self.out(out)
