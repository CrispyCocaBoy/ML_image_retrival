import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from config import ModelConfig

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class ResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.3, batch_norm=True, freeze_backbone=False):
        super().__init__()

        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.pooling = GeM()

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("layer4"):
                    param.requires_grad = False

        layers = [nn.Linear(2048, 512)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(512, embedding_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))

        self.embedding = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.pooling(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def resnet50(model_cfg: ModelConfig, pretrained=True):
    model = ResNetEmbedder(
        embedding_dim=model_cfg.embedding_dim,
        dropout=model_cfg.dropout,
        batch_norm=model_cfg.batch_norm,
        freeze_backbone=model_cfg.freeze_backbone
    )

    if not pretrained:
        for param in model.backbone.parameters():
            if param.requires_grad and len(param.shape) >= 2:
                nn.init.kaiming_normal_(param.data, nonlinearity="relu")
            elif param.requires_grad:
                nn.init.zeros_(param.data)

    return model
