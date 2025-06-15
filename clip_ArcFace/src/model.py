import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class ViTClassifier(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        freeze: bool = True
    ):
        super().__init__()
        self.device = device

        # Carica il modello CLIP con visual.proj incluso
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)
        if freeze:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Usa la dimensione dell'embedding prodotto da CLIP (es. 512)
        embed_dim = self.clip_model.visual.output_dim

        # FC per ArcFace o Cosine Classifier (senza bias, normalizzato)
        self.margin_fc = nn.Linear(embed_dim, num_classes, bias=False).to(device)
        nn.init.xavier_uniform_(self.margin_fc.weight)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        # Ottieni direttamente l'embedding da CLIP (include visual.proj)
        with torch.no_grad():
            emb = self.clip_model.encode_image(x.to(self.device)).float()

        emb = F.normalize(emb, dim=1)

        if not return_logits:
            return emb

        weight = F.normalize(self.margin_fc.weight, dim=1)
        logits = F.linear(emb, weight)
        return logits

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.clip_model.encode_image(x.to(self.device)).float()
        return F.normalize(emb, dim=1)
