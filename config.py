from dataclasses import dataclass

@dataclass
class DataConfig:
    train_data_root: str = "data_example_animal/train"
    query_data_root: str = "data_example_animal/test/query"
    gallery_data_root: str = "data_example_animal/test/gallery"

@dataclass
class ModelConfig:
    embedding_dim: int = 128                # Dimension of the embedding
    backbone_type: str = "resnet50"         # Kind of backbone
    dropout: float = 0.0                    # Dropout on the head of the embedding
    batch_norm: bool = True                 # BatchNorm after backbone
    freeze_backbone: bool = False           # If True, freezes the backbone

@dataclass
class TrainingConfig:
    optimizer: str = "adam"                 # "adam" o "sgd"
    learning_rate: float = 1e-4             # Learning rate
    weight_decay: float = 0                 # L2 regularization
    loss: str = "triplet"                   # Kind di loss (triplet, contrastive, ecc.)
    margin: float = 0.2                     # Margin for triplet loss
    epochs: int = 10                        # Number of epochs
    batch_size: int = 32                    # Batch dimension
    mining_strategy: str = "semi-hard"      # "random" or "semi-hard"
    compiled: bool = False                   # If True, use torch.compile
    model_save_path: str = "model_repository/resnet_triplet.pth"  # Where to save the model
