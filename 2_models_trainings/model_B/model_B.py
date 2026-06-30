"""
model_B.py — Model B: ResNet-50 + Shadow Mask (4-channel input)

The only architectural difference from Model A is the first convolutional
layer, which is modified from 3 to 4 input channels.

Strategy for the 4th channel weights (standard approach):
    The pretrained ResNet-50 first conv expects 3 channels.
    We create a new 4-channel conv and:
        - Copy the 3 pretrained channel weights exactly (preserve ImageNet knowledge)
        - Initialise the 4th channel as the mean of the 3 pretrained channels
          (a neutral starting point — the network can learn to use the mask
          from there without random noise disrupting early training)

Everything else — backbone stages, GAP, classification head, two-stage
fine-tuning — is identical to Model A.

Architecture:
    4-ch input [B, 4, 224, 224]
    └── ResNet-50 backbone (modified first conv: 3→4 ch)
        └── GAP → [B, 2048]
            └── FC(512) → ReLU → Dropout(0.5) → FC(1)
                └── Sigmoid at inference / BCEWithLogitsLoss at training
"""

import torch
import torch.nn as nn
from torchvision import models


class ModelB(nn.Module):

    def __init__(self, fc_size: int = 512, dropout: float = 0.5):
        super().__init__()

        # Load pretrained ResNet-50 
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Modify first conv layer: 3 → 4 input channels
        old_conv = resnet.conv1
        # Create new conv with 4 input channels, same everything else
        new_conv = nn.Conv2d(
            in_channels  = 4,
            out_channels = old_conv.out_channels,
            kernel_size  = old_conv.kernel_size,
            stride       = old_conv.stride,
            padding      = old_conv.padding,
            bias         = False,
        )
        # Copy pretrained weights for the first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        resnet.conv1 = new_conv

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output: [B, 2048, 1, 1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, fc_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, 224, 224]  — RGB + shadow mask, normalised
        Returns:
            logits: [B, 1]
        """
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, lr_backbone: float, lr_head: float) -> list:
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(),     "lr": lr_head},
        ]

    def count_parameters(self) -> dict:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params     = sum(p.numel() for p in self.head.parameters())
        trainable       = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone":  backbone_params,
            "head":      head_params,
            "total":     backbone_params + head_params,
            "trainable": trainable,
        }
