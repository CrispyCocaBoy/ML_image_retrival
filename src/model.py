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

# Resnet50V2

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class ResNetEmbedderV2(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=False):
        super().__init__()

        # 1. Carica ResNet50 pre-addestrata su ImageNet
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # 2. Rimuovi l'avgpool e il classificatore
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # 3. Aggiungi GeM pooling per una estrazione più flessibile
        self.gem = GeM()

        # 4. Freeze opzionale dei primi layer del backbone
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("layer4"):  # Solo l'ultimo blocco rimane aggiornabile
                    param.requires_grad = False

        # 5. Testa di proiezione (projection head)
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.gem(x)                # GeM pooling al posto di avgpool
        x = self.embedding(x)         # Proiezione nello spazio degli embedding
        x = F.normalize(x, p=2, dim=1) # L2-normalizzazione finale
        return x

def resnet50v2(embedding_dim=128, pretrained=True, freeze_backbone=False):
    model = ResNetEmbedderV2(embedding_dim=embedding_dim, freeze_backbone=freeze_backbone)

    if not pretrained:
        for param in model.backbone.parameters():
            if param.requires_grad and len(param.shape) >= 2:
                nn.init.kaiming_normal_(param.data, nonlinearity="relu")
            elif param.requires_grad:
                nn.init.zeros_(param.data)
