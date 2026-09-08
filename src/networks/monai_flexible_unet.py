"""MONAI FlexibleUNet baselines with ImageNet-pretrained EfficientNet encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class MonaiFlexibleUNet(nn.Module):
    """2D MONAI FlexibleUNet with ImageNet preprocessing inside the model."""

    def __init__(self, backbone: str, num_classes: int, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        if in_channels != 3:
            raise ValueError("ImageNet-pretrained MONAI FlexibleUNet requires 3-channel RGB input.")
        try:
            from monai.networks.nets import FlexibleUNet
        except ImportError as error:
            raise ImportError(
                "Install MONAI in this environment with `pip install monai`."
            ) from error
        self.model = FlexibleUNet(
            in_channels=in_channels,
            out_channels=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            spatial_dims=2,
            is_pad=True,
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model((image - self.image_mean) / self.image_std)
