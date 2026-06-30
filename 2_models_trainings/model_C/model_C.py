"""
model_C.py — Model C: Physics-Guided Dual-Branch Shadow Comparison

Architecture:
    Branch 1 (observed shadow):  small CNN → GAP → 256-d features
    Branch 2 (expected shadow):  small CNN → GAP → 256-d features
    Concatenation:               512-d combined features
    Head:                        FC(256) → ReLU → Dropout(0.5) → FC(1)

Both CNNs are identical in structure but have independent weights —
they learn separately what observed shadows mean vs what expected
shadows mean, then the head learns their relationship.

No pretrained weights. Everything trains from scratch on QQB.
Single training stage, 30 epochs.

Why small CNNs and not ResNet-50?
    ResNet-50 is pretrained on RGB ImageNet images. Our inputs are
    single-channel shadow maps with very different statistics.
    A small CNN trained from scratch on our specific signals is
    more appropriate and avoids the mismatch.
"""

import torch
import torch.nn as nn


class ShadowCNN(nn.Module):
    """
    Small CNN for processing a single-channel shadow map.

    Architecture:
        Conv(1→32)  + BN + ReLU + MaxPool   →  112×112×32
        Conv(32→64) + BN + ReLU + MaxPool   →   56×56×64
        Conv(64→128)+ BN + ReLU + MaxPool   →   28×28×128
        Conv(128→feature_dim) + BN + ReLU   →   28×28×feature_dim
        GlobalAveragePool                   →   feature_dim

    BatchNorm is important here: shadow maps have very different
    intensity distributions between images, BN stabilises training.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 56 → 28

            # Block 4 — compress to feature_dim
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 224, 224]
        Returns:
            features: [B, feature_dim]
        """
        x = self.features(x)       # [B, feature_dim, 28, 28]
        x = self.gap(x)            # [B, feature_dim, 1, 1]
        return x.flatten(1)        # [B, feature_dim]


class ModelC(nn.Module):
    """
    Dual-branch shadow comparison model.

    Branch 1 processes the observed shadow mask (what shadows are there).
    Branch 2 processes the expected shadow map (what shadows should be there).
    The classification head learns their relationship.
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim

        self.branch_observed = ShadowCNN(feature_dim)
        self.branch_expected = ShadowCNN(feature_dim)

        # Classification head
        combined_dim = feature_dim * 2
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        observed: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            observed: [B, 1, 224, 224] — observed shadow mask
            expected: [B, 1, 224, 224] — expected shadow coherence map
        Returns:
            logits: [B, 1]
        """
        feat_obs = self.branch_observed(observed)   # [B, 256]
        feat_exp = self.branch_expected(expected)   # [B, 256]
        combined = torch.cat([feat_obs, feat_exp], dim=1)  # [B, 512]
        return self.head(combined)                  # [B, 1]

    @torch.no_grad()
    def predict(
        self,
        observed: torch.Tensor,
        expected: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        self.eval()
        return (torch.sigmoid(self.forward(observed, expected)) >= threshold).long()

    def count_parameters(self) -> dict:
        branch_obs = sum(p.numel() for p in self.branch_observed.parameters())
        branch_exp = sum(p.numel() for p in self.branch_expected.parameters())
        head       = sum(p.numel() for p in self.head.parameters())
        total      = branch_obs + branch_exp + head
        return {
            "branch_observed": branch_obs,
            "branch_expected": branch_exp,
            "head":            head,
            "total":           total,
        }
