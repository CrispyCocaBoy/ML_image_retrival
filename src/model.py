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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

# GeM Pooling
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

# Embedder con ResNet50 modificata
class ResNetEmbedderV2(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=False):
        super().__init__()

        # 1. Backbone ResNet50
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # 2. Rimuove avgpool e classificatore
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # 3. GeM pooling
        self.gem = GeM()

        # 4. Freezing opzionale dei layer tranne layer4
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("layer4"):
                    param.requires_grad = False

        # 5. Testa di embedding
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        # Forward "manuale" per inserire GeM
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.gem(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x

# Funzione per costruire il modello
def resnet50v2(model_cfg, pretrained=True):
    return EmbeddingNet(
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

# GeM Pooling
class GeM2(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

# Embedder migliorato
class ResNetEmbedderV3(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=False):
        super().__init__()

        # Backbone
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # GeM Pooling
        self.gem = GeM2()

        # Freezing
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("layer4"):
                    param.requires_grad = False

        # Testa embedding migliorata
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim, bias=False),
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

        x = self.gem(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x
def resnet50v3(model_cfg, pretrained=True):
    return EmbeddingNet(
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )