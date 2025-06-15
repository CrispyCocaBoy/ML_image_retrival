from types import SimpleNamespace
import torch

config = SimpleNamespace(
    train_dir="../data/train",
    validation_dir="../data/validation",
    query_dir="../data/test/query",
    gallery_dir="../data/test/gallery",
    batch_size=64,
    epochs=35,
    learning_rate=1e-4,
    embedding_dim=1024,
    top_k=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    distance_metric="cosine",
    force_train=True,
    dropout=0,
    weight_decay=1e-4,
    margin=0.2,
    save_retrieval_results_per_epoch=True,
    compute_validation_loss=True
)
