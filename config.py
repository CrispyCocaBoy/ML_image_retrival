from dataclasses import dataclass

@dataclass
class DataConfig:
    train_data_root: str = "data_example_animal/train"
    query_data_root: str = "data_example_animal/test/query"
    gallery_data_root: str = "data_example_animal/test/gallery"

@dataclass
class ModelConfig:
    embedding_dim: int = 128                # Dimensione dello spazio embedding
    backbone_type: str = "resnet50"         # Tipo di backbone, es. "resnet18", "resnet50", "resnet101"
    dropout: float = 0.0                    # Dropout nella testa di embedding
    batch_norm: bool = True                 # BatchNorm dopo il backbone
    freeze_backbone: bool = False           # Se True, congela il backbone

@dataclass
class TrainingConfig:
    optimizer: str = "adam"                 # "adam" o "sgd"
    learning_rate: float = 1e-4             # Learning rate
    weight_decay: float = 0              # L2 regularization
    loss: str = "triplet"                   # Tipo di loss (triplet, contrastive, ecc.)
    margin: float = 0.2                     # Margine per triplet loss
    epochs: int = 10                        # Numero di epoche
    batch_size: int = 32                    # Dimensione del batch
    mining_strategy: str = "semi-hard"      # "random" o "semi-hard"
    compiled: bool = False                   # Se True, usa torch.compile
    model_save_path: str = "model_repository/resnet_triplet.pth"  # Dove salvare il modello
    distance_metric: str = "euclidean"      # oppure "cosine"