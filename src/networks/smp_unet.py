"""ImageNet-pretrained U-Net baselines provided by segmentation-models-pytorch."""

from __future__ import annotations

import torch
import torch.nn as nn


class SMPUnet(nn.Module):
    """U-Net wrapper that keeps ImageNet normalization inside the model."""

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        decoder_attention_type: str | None = None,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("ImageNet-pretrained SMP U-Net baselines require 3-channel RGB input.")
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "Install the optional ResNet U-Net dependency with "
                "`pip install -r requirements.txt`."
            ) from error
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_attention_type=decoder_attention_type,
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def encoder(self):
        """Expose the SMP encoder to the shared freeze/unfreeze trainer logic."""
        return self.model.encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.image_mean) / self.image_std
        return self.model(image)
