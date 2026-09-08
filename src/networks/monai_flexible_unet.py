"""MONAI FlexibleUNet baselines with ImageNet-pretrained EfficientNet encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # EfficientNet feature stages downsample by 32.  Older MONAI
        # FlexibleUNet releases can misalign decoder skips for non-multiples
        # of 32 (for example the shared 720x720 training resolution). Pad in
        # raw image space, then crop logits back to the caller's resolution.
        height, width = image.shape[-2:]
        padded_height = ((height + 31) // 32) * 32
        padded_width = ((width + 31) // 32) * 32
        if (padded_height, padded_width) != (height, width):
            image = F.pad(image, (0, padded_width - width, 0, padded_height - height))
        logits = self.model((image - self.image_mean) / self.image_std)
        return logits[..., :height, :width]
