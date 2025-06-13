import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class ViTClassifier(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        embed_dim: int = 128,
        freeze: bool = True
    ):
        super().__init__()
        self.device = device

        # Carica il modello CLIP e congela se richiesto
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)
        if freeze:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        visual_dim = self.clip_model.visual.output_dim

        # Proiettore con dropout (stessa struttura del modello originale)
        self.projection = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.Dropout(0.3)
        ).to(device)

        # ArcFace fully-connected: embed_dim -> num_classes
        self.margin_fc = nn.Linear(embed_dim, num_classes, bias=False).to(device)
        nn.init.xavier_uniform_(self.margin_fc.weight)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        # Estrai feature CLIP senza grad
        with torch.no_grad():
            feat_clip = self.clip_model.encode_image(x.to(self.device))

        # Proietta e normalizza
        emb = self.projection(feat_clip.float())
        emb = F.normalize(emb, dim=1)

        # Se serve solo embeddings per inferenza
        if not return_logits:
            return emb

        # Altrimenti calcola cosine logits per ArcFace
        weight = F.normalize(self.margin_fc.weight, dim=1)
        logits = F.linear(emb, weight)
        return logits

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # Metodo per inferenza embedding puro
        with torch.no_grad():
            feat_clip = self.clip_model.encode_image(x.to(self.device))
        emb = self.projection(feat_clip.float())
        return F.normalize(emb, dim=1)
