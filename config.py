import torch
from types import SimpleNamespace

# Define configuration parameters
config = SimpleNamespace(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Dataset and Dataloader settings
    train_dir="Competition/train",  # Corrected path based on previous discussion
    query_dir="Competition/test/query",
    gallery_dir="Competition/test/gallery",
    
    # Model and Training settings
    embedding_dim=512,         # Dimension of the projected embeddings
    batch_size=64,             # Batch size for training and inference
    epochs=2,                 # Recommended starting point, increase later if needed
    learning_rate=1e-4,        # Learning rate for the NEW projection layers (higher)
    weight_decay=1e-4,         # Weight decay for regularization (L2 penalty)
    dropout_rate=0.3,          # Dropout rate for projection layers
    
    # --- Control training behavior ---
    force_train=True,          # <--- ADDED/ENSURED THIS IS PRESENT: If True, always retrain; otherwise, loads if finetuned_clip.pth exists.
    freeze_clip=False,         # Set to False to unfreeze the CLIP backbone
    # Ratio for CLIP backbone learning rate: backbone_lr = learning_rate * ratio
    # Typically, a very small value like 0.01 or 0.1 compared to the projection layer LR.
    clip_backbone_learning_rate_ratio=0.01, 
    
    # Retrieval settings
    top_k=10,                   # Number of top results to retrieve
    distance_metric="cosine",  # Metric for similarity ("cosine" or "euclidean")
)
