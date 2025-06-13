from types import SimpleNamespace
import torch

config = SimpleNamespace(
    train_dir="Competition/train",
    query_dir="Competition/test/query",
    gallery_dir="Competition/test/gallery",
    batch_size=32,
    epochs=15,
    learning_rate=1e-4,
    embedding_dim=1024,
    top_k=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    distance_metric="cosine",       #"euclidean" or "cosine"
    force_train=True,  # ⬅️ nuovo flag
    dropout=0.3,       # ⬅️ nuovo parametro
    weight_decay=1e-4  # ⬅️ nuovo parametro
)
