import torch
import torch.nn as nn
import clip
from config import config  # aggiungiamo questa importazione

class FineTunedCLIP(nn.Module):
    def __init__(self, device, embed_dim=128, freeze_clip=True):
        super().__init__()
        self.device = device

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Proiettore con dropout preso dalla config
        self.projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, embed_dim),
            nn.Dropout(config.dropout)
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x.to(self.device))
        return self.projection(x.float())

    def encode_image(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x.to(self.device))
        return self.projection(x.float())
