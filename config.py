import torch
from types import SimpleNamespace

config = SimpleNamespace(
    train_dir="Competition/train",
    query_dir="Competition/test/query",
    gallery_dir="Competition/test/gallery",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    distance_metric="cosine"  # scegli tra "cosine" o "l2"
)
