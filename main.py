import torch
import torch.nn as nn
import os
import re
import clip 
import numpy as np 
import random 
from torch.utils.data import DataLoader, Subset # <-- ADDED Subset for train/val split

# Importing config and your custom modules
from config import config
from src.image_text_dataset import ImageTextDataset # <-- CORRECTED IMPORT
from src.finetuned_clip import FineTunedCLIP
from src.training_loop import CLIPLoss, train_clip_hybrid 
from src.evaluation import perform_retrieval_and_save_results # <-- CORRECTED IMPORT (assuming refactored)
from src.results import get_top_k, load_images_from_folder # This was already correct based on your files

# --- NEW: Function to set all random seeds for reproducibility ---
def set_seed(seed):
    """
    Sets the random seed for reproducibility across multiple libraries.
    
    Args:
        seed (int): The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
    print(f"✅ Random seed set to {seed} for reproducibility.")

# --- Main execution logic ---
def check_and_train_model():
    """
    Checks for existing checkpoints and either resumes training or starts new training.
    Handles model loading, training, and evaluation.
    """
    # Load configuration
    cfg = config 
    device = cfg.device

    # --- NEW: Set seed at the very beginning of the training run ---
    set_seed(cfg.seed)
    # --- End seed setting ---

    # Setup model directories
    checkpoints_dir = os.path.join("repository", "checkpoints")
    all_weights_dir = os.path.join("repository", "all_weights")
    metrics_dir = os.path.join("repository", "metrics")
    retrieval_results_dir = os.path.join("repository", "results") # <-- Changed to 'results' to match your folder
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(all_weights_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(retrieval_results_dir, exist_ok=True) # Ensure this exists for evaluation results

    latest_checkpoint_path = None
    start_epoch = 0
    model = None # Initialize model variable

    # Load the pre-trained CLIP model (once)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Initialize your FineTunedCLIP model (our custom model on top of CLIP)
    initial_model = FineTunedCLIP(
        device,
        cfg.embedding_dim,
        freeze_clip=cfg.freeze_clip 
    ).to(device)


    if not cfg.force_train:
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith('weights_epoch_') and f.endswith('.pth')]
        if checkpoints:
            epochs_found = []
            for f in checkpoints:
                match = re.search(r'weights_epoch_(\d+)\.pth', f)
                if match:
                    epochs_found.append(int(match.group(1)))
            
            if epochs_found:
                latest_epoch = max(epochs_found)
                latest_checkpoint_path = os.path.join(checkpoints_dir, f"weights_epoch_{latest_epoch}.pth")
                start_epoch = latest_epoch # Start training from the epoch AFTER this checkpoint
                
                try:
                    initial_model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
                    model = initial_model
                    print(f"✅ Found latest checkpoint: {latest_checkpoint_path}, resuming from epoch {start_epoch + 1}.")
                except Exception as e:
                    print(f"❌ Error loading checkpoint {latest_checkpoint_path}: {e}. Starting new training.")
                    model = initial_model # Fallback to initial model if load fails
                    start_epoch = 0 # Reset start epoch
            else:
                print("No valid checkpoints found. Starting new training.")
                model = initial_model
        else:
            print("No checkpoints found. Starting new training.")
            model = initial_model
    else:
        print("⚠️ config.force_train is True. Starting new training despite existing checkpoints.")
        model = initial_model # Use the freshly initialized model

    # Create datasets and dataloaders
    # Use ImageTextDataset from src/image_text_dataset.py
    full_train_dataset = ImageTextDataset(cfg.train_dir, preprocess)

    # --- Split dataset into training and validation sets ---
    dataset_size = len(full_train_dataset)
    # Using 15% for validation, adjust if needed.
    validation_split = 0.15 
    indices = list(range(dataset_size))
    random.shuffle(indices) 
    
    split = int(validation_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices) # This uses your ImageTextDataset and preprocess

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Train the model
    print("\n--- Starting Model Training ---")
    trained_model = train_clip_hybrid(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
    )
    print("\n--- Training Complete ---")

    # After training, perform final evaluation with the trained_model
    print("\n--- Starting Final Model Evaluation ---")
    
    # We will pass the `epochs` value as `epoch_number` for naming the final result file.
    # This is slightly simplified; in a real scenario, you might evaluate the best epoch
    # based on validation loss, not necessarily the last one trained.
    final_epoch_for_results = cfg.epochs 
    if start_epoch > 0: 
        final_epoch_for_results = start_epoch + (cfg.epochs - start_epoch) 

    # --- Call the refactored evaluation function ---
    perform_retrieval_and_save_results(
        model=trained_model,
        query_dir=cfg.query_dir,
        gallery_dir=cfg.gallery_dir,
        top_k=cfg.top_k,
        distance_metric=cfg.distance_metric,
        device=device,
        batch_size=cfg.batch_size, # Pass batch_size for embedding extraction
        epoch_number=final_epoch_for_results 
    )
    print("\n--- Final Model Evaluation Complete ---")

if __name__ == "__main__":
    check_and_train_model()
