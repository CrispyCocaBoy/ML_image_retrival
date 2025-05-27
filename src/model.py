import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, backbone_type="resnet50", embedding_dim=128, dropout=0.0, batch_norm=True, freeze_backbone=False):
        super().__init__()

        # === BACKBONE SELECTION ===
        if backbone_type == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = 512
        elif backbone_type == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = 2048
        elif backbone_type == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Rimuove FC

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # === HEAD ===
        layers = [nn.Linear(in_features, embedding_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)           # (B, C, 1, 1)
        features = torch.flatten(features, 1) # (B, C)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)  # Normalized

# Factory method
def build_model(model_cfg, pretrained=True):
    return EmbeddingNet(
        backbone_type=model_cfg.backbone_type,
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )
