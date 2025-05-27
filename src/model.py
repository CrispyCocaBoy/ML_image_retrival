import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.0, batch_norm=True, freeze_backbone=False):
        super().__init__()

        # === BACKBONE ===
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # removes final fully connected classification layer

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # === HEAD ===
        layers = [nn.Linear(backbone.fc.in_features, embedding_dim)] # Takes each 2048D vector output by the ResNet backbone, passes it through a linear layer, outputs a 128D vector
        if batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.head = nn.Sequential(*layers) # Sequential chains together layers, instead of defining them in different rows you put them inside Sequential()
        # *layers unpack the list into positional arguments

    def forward(self, x):
        features = self.backbone(x)                     # (B, 2048, 1, 1)
        features = torch.flatten(features, 1)           # (B, 2048)
        # for each image in the batch, you have a 2048-dimensional vector (the extracted features)
        embeddings = self.head(features)                # (B, embedding_dim)
        # This head transforms each 2048D vector into a lower-dimensional embedding vector, typically (B, embedding_dim) where embedding_dim is something like 128
        return F.normalize(embeddings, p=2, dim=1)      # normalized

def resnet50(model_cfg, pretrained=True):
    return EmbeddingNet(
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )
