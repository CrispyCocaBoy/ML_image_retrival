import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
import torch

class ResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Carica ResNet-50 pre-addestrata
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features

        # Rimuove il classificatore originale
        self.backbone.fc = nn.Identity()

        # Proiezione nello spazio degli embedding
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)           # output: [batch_size, 2048]
        x = self.embedding(x)          # output: [batch_size, embedding_dim]
        x = F.normalize(x, p=2, dim=1) # L2-normalizzazione
        return x


def resnet50(pretrained=True, embedding_dim=128):
    """
    Factory function per creare un ResNetEmbedder con o senza pesi ImageNet.
    """
    model = ResNetEmbedder(embedding_dim=embedding_dim)
    if not pretrained:
        # Reset dei pesi del backbone (se non vuoi ImageNet)
        for param in model.backbone.parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Solo per matrici di peso
                nn.init.kaiming_normal_(param.data, nonlinearity="relu")
            elif param.requires_grad:  # Per bias e altri parametri 1D
                nn.init.zeros_(param.data)
    return model

class ResNetEmbedderV2(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Carica ResNet-50 pre-addestrata
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features

        # Rimuove il classificatore originale
        self.backbone.fc = nn.Identity()

        # Proiezione nello spazio degli embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)  # opzionale, ma utile per stabilizzare
        )

    def forward(self, x):
        x = self.backbone(x)           # output: [batch_size, 2048]
        x = self.embedding(x)          # output: [batch_size, embedding_dim]
        x = F.normalize(x, p=2, dim=1) # L2-normalizzazione
        return x


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

    return model


