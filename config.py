from types import SimpleNamespace
import torch

config = SimpleNamespace(
    train_dir="Competition/train",
    query_dir="Competition/test/query",
    gallery_dir="Competition/test/gallery",
    batch_size=32,
    epochs=5,
    learning_rate=1e-3,
    embedding_dim=256,
    top_k=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    distance_metric="euclidean",       #"euclidean" or "cosine"
    force_train=True  # ⬅️ nuovo flag
)
