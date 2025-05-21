from dataclasses import dataclass

@dataclass
class ModelConfig:
    embedding_dim: int = 128                # Dimensione dello spazio embedding
    backbone_type: str = "resnet50"         # Tipo di backbone, es. "resnet18", "resnet50"
    dropout: float = 0.0                    # Dropout nella testa di embedding
    batch_norm: bool = True                 # BatchNorm dopo il backbone
    freeze_backbone: bool = False           # Se True, congela il backbone

@dataclass
class TrainingConfig:
    optimizer: str = "adam"                 # "adam" o "sgd"
    learning_rate: float = 1e-4             # Learning rate
    weight_decay: float = 1e-5              # L2 regularization
    loss: str = "triplet"                   # Tipo di loss (triplet, contrastive, ecc.)
    margin: float = 0.1                     # Margine per triplet loss
    epochs: int = 2                        # Numero di epoche
    batch_size: int = 32                    # Dimensione del batch
    mining_strategy: str = "random"      # "random" o "semi-hard"
    compiled: bool = True                   # Se True, usa torch.compile
    model_save_path: str = "model_repository/resnet_triplet.pth"  # Dove salvare il modello
