import torch
import torch.nn as nn
import clip

class SiameseNetworkCLIP(nn.Module):
    def __init__(self, device, embed_dim=64, freeze_clip=True):
        super().__init__()
        self.device = device

        # Carica il modello CLIP e portalo sul device
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        # Congela i pesi del backbone CLIP se richiesto
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Proiettore per abbassare la dimensionalità degli embedding
        self.projector = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        ).to(device)

    def forward_once(self, x):
        # Encode image (usa no_grad se CLIP è freezato)
        with torch.no_grad() if not any(p.requires_grad for p in self.clip_model.parameters()) else torch.enable_grad():
            x = self.clip_model.encode_image(x.to(self.device))
        return self.projector(x.float())

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        return feat1, feat2
