import torch.nn as nn
import torch.nn.functional as F
import torch

class EmbeddingNet(nn.Module):
    def __init__(self, backbone, feature_dim, embedding_dim=128, dropout=0.0, batch_norm=True, freeze_backbone=False):
        super().__init__()
        
        self.backbone = backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier head with custom embedding head
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(feature_dim, embedding_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)                     # Shape: (B, feature_dim, 1, 1)
        features = torch.flatten(features, 1)           # Shape: (B, feature_dim)
        embeddings = self.head(features)                # Shape: (B, embedding_dim)
        return F.normalize(embeddings, p=2, dim=1)
    
def resnet50(model_cfg, pretrained=True):
    return EmbeddingNet(
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )