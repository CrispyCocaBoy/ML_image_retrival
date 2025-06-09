# src/efficientnet_embedder.py
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetEmbedder(nn.Module):
    def __init__(self, embed_dim=128, freeze_backbone=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classifier with a linear projection to embed_dim
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.backbone(x)
