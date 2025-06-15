import torch
import torch.nn as nn
import clip

class SiameseNetwork(nn.Module):
    def __init__(self, device, embed_dim=128, freeze=True):
        super().__init__()
        self.device = device

        # Carica CLIP e congela se richiesto
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Proiettore lineare + dropout
        self.projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, embed_dim),
            nn.Dropout(0.3)
        ).to(device)

    def encode_image(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x.to(self.device))
        return self.projection(x.float())

    def forward(self, x1, x2=None):
        e1 = self.encode_image(x1)
        if x2 is None:
            return e1  # [B, embed_dim]
        e2 = self.encode_image(x2)
        return e1, e2
