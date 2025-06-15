import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import csv
import re 

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

        # Create labels for cross-entropy
        labels = torch.arange(len(image_features), device=image_features.device)

        # Calculate the total loss as the average of two cross-entropy losses:
        # 1. Image-to-text matching (predicting which text matches each image)
        # 2. Text-to-image matching (predicting which image matches each text)
        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss

def train_clip_hybrid(model, train_dataloader, val_dataloader, device="cuda", 
                      epochs=10, lr=1e-5, start_epoch: int = 0):
    """
    Trains a given PyTorch model using the CLIP-style hybrid (contrastive + cross-entropy) loss.

    Args:
        model (torch.nn.Module): The neural network model to be trained (e.g., FineTunedCLIP).
                                 It should have methods `encode_image` and `encode_text`
                                 and expose a `logit_scale` parameter (e.g., via a property).
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation data.
        device (str, optional): The device ('cuda' or 'cpu') to run the
                                training on. Defaults to "cuda".
        epochs (int, optional): The total number of training epochs to run (exclusive end).
                                Training will run from `start_epoch` up to `epochs`.
                                Defaults to 10.
        lr (float, optional): The learning rate for the Adam optimizer.
                              Defaults to 1e-5.
        start_epoch (int, optional): The epoch number to start training from.
                                     Useful for resuming training. Defaults to 0.

    Returns:
        Tuple[torch.nn.Module, List[str]]: A tuple containing:
            - torch.nn.Module: The final trained model.
            - List[str]: A list of file paths to all saved full model checkpoints (.pt files).
    """
    model = model.to(device)
    
    # Implement Discriminative Learning Rates
    optimizer_params = []
    clip_backbone_params = []
    projection_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip_model' in name:
                clip_backbone_params.append(param)
            else:
                projection_params.append(param)
    
    if projection_params:
        optimizer_params.append({'params': projection_params, 'lr': lr})
    
    if clip_backbone_params:
        backbone_lr = lr * config.clip_backbone_learning_rate_ratio
        optimizer_params.append({'params': clip_backbone_params, 'lr': backbone_lr})

    optimizer = optim.Adam(optimizer_params, betas=(0.9, 0.98), eps=1e-6, weight_decay=config.weight_decay)

    criterion = CLIPLoss()

    # Setup directories for saving metrics and weights
    metrics_dir = os.path.join("repository", "metrics")
    all_weights_dir = os.path.join("repository", "all_weights")
    checkpoints_dir = os.path.join("repository", "checkpoints")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(all_weights_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    metrics_file_path = os.path.join(metrics_dir, "training_metrics.csv")
    
    # Logic for appending or overwriting CSV based on start_epoch
    if start_epoch > 0 and os.path.exists(metrics_file_path):
        metrics_file_mode = 'a' # Append mode
    else:
        metrics_file_mode = 'w' # Write mode (overwrite)
        
    with open(metrics_file_path, metrics_file_mode, newline='') as csvfile:
        metric_writer = csv.writer(csvfile)
        if metrics_file_mode == 'w': # Write header only if overwriting
            metric_writer.writerow(['epoch', 'train_loss', 'val_loss'])

    saved_full_model_paths = []

    # --- Training Loop ---
    for epoch in range(start_epoch, epochs):
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Validation)") 
        with torch.no_grad():
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
        # Save entire model (full model object)
        full_model_path = os.path.join(all_weights_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model, full_model_path)
        print(f"Model saved to {full_model_path}")
        saved_full_model_paths.append(full_model_path) # Add to list

        # Save only model's state_dict
        state_dict_path = os.path.join(checkpoints_dir, f"weights_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), state_dict_path)
        print(f"Weights saved to {state_dict_path}")

    return model, saved_full_model_paths 
