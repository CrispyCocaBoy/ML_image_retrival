import torch
from types import SimpleNamespace

config = SimpleNamespace(
    train_dir="Competition/train",  # eventualmente inutilizzato con CLIP
    query_dir="Competition/test/query",
    gallery_dir="Competition/test/gallery",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
