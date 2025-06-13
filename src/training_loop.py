import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os # Import os for path operations and directory creation
import csv # Import csv for logging metrics

# Importing actual config and FineTunedCLIP from your project structure
from config import config
from src.finetuned_clip import FineTunedCLIP

class CLIPLoss(nn.Module):
    """
    Implements the Contrastive Language-Image Pre-training (CLIP) loss.

    This loss function calculates symmetric cross-entropy between image and text
    embeddings based on their cosine similarity. It requires a learnable
    temperature parameter (logit_scale) from the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # Normalize features to prepare for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between all image and text features in the batch.
        # The result is a square matrix where element (i, j) represents the similarity
        # between the i-th image embedding and the j-th text embedding.
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # Create labels for cross-entropy: the diagonal elements correspond to the
        # correct (positive) pairs. For a batch of N items, the labels are [0, 1, ..., N-1].
        labels = torch.arange(len(image_features), device=image_features.device)

        # Calculate the total loss as the average of two cross-entropy losses:
        # 1. Image-to-text matching (predicting which text matches each image)
        # 2. Text-to-image matching (predicting which image matches each text)
        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss

def train_clip_hybrid(model, train_dataloader, val_dataloader, device="cuda", epochs=10, lr=1e-5):
    """
    Trains a given PyTorch model using the CLIP-style hybrid (contrastive + cross-entropy) loss.

    This function sets up the model for training, defines an Adam optimizer
    and a CLIPLoss criterion. It then iterates through the specified number
    of epochs, processing image-text pairs from the dataloader. For each batch,
    it encodes both images and texts, computes the CLIP loss, performs
    backpropagation, and updates the model's weights and the learnable temperature.
    Training progress and loss are displayed using tqdm.

    Args:
        model (torch.nn.Module): The neural network model to be trained (e.g., FineTunedCLIP).
                                 It should have methods `encode_image` and `encode_text`
                                 and expose a `logit_scale` parameter (e.g., via a property).
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation data.
        device (str, optional): The device ('cuda' or 'cpu') to run the
                                training on. Defaults to "cuda".
        epochs (int, optional): The number of training epochs. Defaults to 10.
        lr (float, optional): The learning rate for the Adam optimizer.
                              Defaults to 1e-5.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    criterion = CLIPLoss()

    # --- Setup directories for saving metrics and weights ---
    metrics_dir = os.path.join("repository", "metrics")
    all_weights_dir = os.path.join("repository", "all_weights")
    checkpoints_dir = os.path.join("repository", "checkpoints")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(all_weights_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    metrics_file_path = os.path.join(metrics_dir, "training_metrics.csv")
    
    # Open CSV file and write header
    with open(metrics_file_path, 'w', newline='') as csvfile:
        metric_writer = csv.writer(csvfile)
        metric_writer.writerow(['epoch', 'train_loss', 'val_loss'])

    # --- Training Loop ---
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        total_train_loss = 0
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Training)")

        for images, texts in train_loop:
            images = images.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            loss = criterion(image_features, text_features, model.logit_scale)
            loss.backward()
            # --- ADDED: Gradient Clipping ---
            # This prevents gradients from exploding, which can lead to NaN loss.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # You can adjust max_norm
            # --- END ADDED ---
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Validation)")
        with torch.no_grad(): # No need to calculate gradients during validation
            for images, texts in val_loop:
                images = images.to(device)
                texts = texts.to(device)

                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)

                loss = criterion(image_features, text_features, model.logit_scale)
                total_val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

        # --- Log metrics to CSV ---
        with open(metrics_file_path, 'a', newline='') as csvfile:
            metric_writer = csv.writer(csvfile)
            metric_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        # --- Save model and weights for current epoch ---
        # Save entire model
        torch.save(model, os.path.join(all_weights_dir, f"model_epoch_{epoch+1}.pt"))
        print(f"✅ Model saved to {os.path.join(all_weights_dir, f'model_epoch_{epoch+1}.pt')}")

        # Save only model's state_dict
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"weights_epoch_{epoch+1}.pth"))
        print(f"✅ Weights saved to {os.path.join(checkpoints_dir, f'weights_epoch_{epoch+1}.pth')}")

    return model
