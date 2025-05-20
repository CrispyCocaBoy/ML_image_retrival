import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

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




