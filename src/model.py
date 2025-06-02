import torch.nn as nn
import torchvision.models as models


def build_model(backbone_name, embedding_dim, num_classes, pretrained=True):
    if backbone_name == "resnet18":
        backbone = models.resnet18(pretrained=pretrained)
    elif backbone_name == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, embedding_dim),
        nn.ReLU(),
        nn.Linear(embedding_dim, num_classes)
    )
    return backbone
