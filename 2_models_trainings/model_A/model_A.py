"""
model.py — Model A: Pure End-to-End ML

Architecture:
    ResNet-50 (ImageNet pretrained)
    └── GAP (built into ResNet-50)
        └── FC(512) → ReLU → Dropout(0.5) → FC(1)
            └── Sigmoid at inference time
                └── Damaged (≥ 0.5) / Intact (< 0.5)

No manual features, no shadow mask, no building mask, no SAR.
The model learns entirely from raw RGB pixels.

Note on Sigmoid:
    We do NOT apply sigmoid inside forward(). Instead we use
    BCEWithLogitsLoss during training, which folds the sigmoid
    in numerically — more stable than applying them separately.
    Sigmoid is only applied at inference time (see predict()).
"""

import torch
import torch.nn as nn
from torchvision import models


class ModelA(nn.Module):

    def __init__(self, fc_size: int = 512, dropout: float = 0.5):
        super().__init__()

        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])        
        self.head = nn.Sequential(
            nn.Flatten(),                   # [B, 2048, 1, 1] → [B, 2048]
            nn.Linear(2048, fc_size),       # [B, 2048] → [B, 512]
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_size, 1),          # [B, 512]  → [B, 1]  (logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]  — batch of RGB images, normalized
        Returns:
            logits: [B, 1]  — raw scores (no sigmoid yet)
        """
        features = self.backbone(x)   # [B, 2048, 1, 1]
        return self.head(features)    # [B, 1]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns binary predictions (0 = intact, 1 = damaged).
        Applies sigmoid that is intentionally omitted during training.
        """
        self.eval()
        probs = torch.sigmoid(self.forward(x))   # [B, 1]
        return (probs >= threshold).long()

    def freeze_backbone(self):
        """Stage 1: stop backbone gradients, train only the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Stage 2: allow the backbone to adapt with a small LR."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        lr_backbone: float,
        lr_head: float,
    ) -> list[dict]:
        """
        Returns two parameter groups with different learning rates,
        used by AdamW in Stage 2.
        Backbone gets a much smaller LR to preserve ImageNet knowledge.
        """
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
