import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.0, batch_norm=True, freeze_backbone=False):
        super().__init__()

        # === BACKBONE ===
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # senza FC finale

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # === HEAD ===
        layers = [nn.Linear(backbone.fc.in_features, embedding_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)                     # (B, 2048, 1, 1)
        features = torch.flatten(features, 1)           # (B, 2048)
        embeddings = self.head(features)                # (B, embedding_dim)
        return F.normalize(embeddings, p=2, dim=1)      # normalized

def resnet50(model_cfg, pretrained=True):
    return EmbeddingNet(
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )
