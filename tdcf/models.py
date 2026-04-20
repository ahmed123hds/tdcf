"""
Backbone classifiers for TDCF experiments on MNIST and CIFAR-100.

The paper recommends testing with CNN, ViT, and SSM backbones to show
that TDCF learns *different* fidelity schedules for different architectures.

MNIST  : MNISTConvNet (~468K), MNISTTinyViT (~120K)
CIFAR-100: CIFAR100ResNet18 (~11M), CIFAR100ViT (~9M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MNISTConvNet(nn.Module):
    """
    Small CNN backbone for MNIST classification.
    ~62K parameters.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),                    # 28 → 14
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),                    # 14 → 7
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PatchEmbedding(nn.Module):
    """Patchify + linear projection for a tiny ViT."""

    def __init__(self, img_size: int = 28, patch_size: int = 7,
                 in_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, 1, 28, 28) → (B, embed_dim, 4, 4) → (B, 16, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class MNISTTinyViT(nn.Module):
    """
    Tiny Vision Transformer for MNIST.
    patch_size=7 → 4×4 = 16 tokens.
    2 Transformer encoder layers, embed_dim=64, 4 heads.
    ~120K parameters.
    """

    def __init__(self, num_classes: int = 10, img_size: int = 28,
                 patch_size: int = 7, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)            # (B, 16, 64)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 17, 64)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])          # classify from [CLS]


# ===========================================================================
# CIFAR-100 Backbones
# ===========================================================================

class ResidualBlock(nn.Module):
    """Standard pre-activation residual block."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class CIFAR100ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-100 (32×32 input, no 7×7 stem).
    ~11M parameters.  Standard backbone for CIFAR benchmarks.
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        # CIFAR stem: 3×3 conv, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class CIFAR100ViT(nn.Module):
    """
    Vision Transformer for CIFAR-100 (32×32 RGB).
    patch_size=4 → 8×8 = 64 tokens.
    6 layers, embed_dim=256, 8 heads.  ~9M parameters.
    """

    def __init__(self, num_classes: int = 100, img_size: int = 32,
                 patch_size: int = 4, embed_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = self.drop(torch.cat([cls, tokens], dim=1) + self.pos_embed)
        tokens = self.norm(self.encoder(tokens))
        return self.head(tokens[:, 0])


# ===========================================================================
# Factory
# ===========================================================================

def make_model(backbone: str, dataset: str = "mnist",
               device: torch.device = None) -> nn.Module:
    """
    Instantiate the right backbone for the given dataset.

    Args:
        backbone: 'cnn' or 'vit'
        dataset:  'mnist' or 'cifar100'
        device:   Target device (model is moved to it).
    Returns:
        nn.Module on `device`.
    """
    if dataset == "mnist":
        m = MNISTConvNet() if backbone == "cnn" else MNISTTinyViT()
    elif dataset == "cifar100":
        m = CIFAR100ResNet18() if backbone == "cnn" else CIFAR100ViT()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if device is not None:
        m = m.to(device)
    return m

