import torch
from types import SimpleNamespace

# Define configuration parameters
config = SimpleNamespace(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Dataset and Dataloader settings
    train_dir="Competition/data/train",  
    query_dir="Competition/data/test/query",
    gallery_dir="Competition/data/test/gallery",
    
    # Model and Training settings
    embedding_dim=512,         # Dimension of the projected embeddings
    batch_size=128,             # Batch size for training and inference
    epochs=40,                 # Recommended starting point, increase later if needed
    learning_rate=1e-4,        # Learning rate for the NEW projection layers 
    weight_decay=1e-4,         # Weight decay for regularization (L2 penalty)
    dropout_rate=0.3,          # Dropout rate for projection layers
    
    # --- Control training behavior ---
    force_train=True,         
    freeze_clip=True,        
    # Ratio for CLIP backbone learning rate: backbone_lr = learning_rate * ratio
    clip_backbone_learning_rate_ratio=0.01, 
    
    # Retrieval settings
    top_k=10,                   # Number of top results to retrieve
    distance_metric="cosine",  # Metric for similarity ("cosine" or "euclidean")
    seed = 42
)
