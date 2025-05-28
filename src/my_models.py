import torch.nn as nn
from torchvision import models
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
    
def build_model(model_cfg, pretrained=True):
    if model_cfg.backbone_type == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone = nn.Sequential(*list(base.children())[:-1])
        feature_dim = 2048

    elif model_cfg.backbone_type == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        backbone = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten()
        )
        feature_dim = base.classifier[1].in_features  # 1280
    
    elif model_name == "convnext_small":
        base = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone = nn.Sequential(*list(base.children())[:-1])
        feature_dim = 768

    else:
        raise ValueError(f"Unsupported model")

    return EmbeddingNet(
        backbone=backbone,
        feature_dim=feature_dim,
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )