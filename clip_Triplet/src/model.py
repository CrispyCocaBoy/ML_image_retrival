import torch
import torch.nn as nn
import clip
from clip_Triplet.config import config

class FineTunedCLIP(nn.Module):
    def __init__(self, device, embed_dim=128, freeze_clip=True):
        super().__init__()
        self.device = device

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, embed_dim),
            nn.Dropout(config.dropout)
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x.to(self.device))
        x = self.projection(x.float())
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def encode_image(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x.to(self.device))
        x = self.projection(x.float())
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x
