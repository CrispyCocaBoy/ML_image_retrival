import torch
import torch.nn as nn
import os
import re
import clip 
import numpy as np 
import random 
from torch.utils.data import DataLoader, Subset 

# Importing config and your custom modules
from config import config
from src.image_text_dataset import ImageTextDataset 
from src.finetuned_clip import FineTunedCLIP
from src.training_loop import CLIPLoss, train_clip_hybrid 
from src.evaluation import perform_retrieval_and_save_results 

# --- Function to set all random seeds for reproducibility ---
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
    print(f"Random seed set to {seed} for reproducibility.")

# --- Main execution logic ---
def check_and_train_model():
    """
    Checks for existing checkpoints and either resumes training or starts new training.
    Handles model loading, training, and evaluation for all saved epochs.
    """
    # Load configuration
    cfg = config 
    device = cfg.device

    # --- Set seed at the very beginning of the training run ---
    set_seed(cfg.seed)
    # --- End seed setting ---

    # Setup model directories
    checkpoints_dir = os.path.join("repository", "checkpoints")
    all_weights_dir = os.path.join("repository", "all_weights")
    metrics_dir = os.path.join("repository", "metrics")
    retrieval_results_dir = os.path.join("repository", "results") 
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(all_weights_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(retrieval_results_dir, exist_ok=True)

    latest_checkpoint_path = None
    start_epoch = 0
    model = None 

    # Load the pre-trained CLIP model (once)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Initialize your FineTunedCLIP model (our custom model on top of CLIP)
    initial_model = FineTunedCLIP(
        device,
        cfg.embedding_dim,
        freeze_clip=cfg.freeze_clip,
        clip_model=clip_model
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
                start_epoch = latest_epoch 
                
                try:
                    initial_model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
                    model = initial_model
                    print(f"Found latest checkpoint: {latest_checkpoint_path}, resuming from epoch {start_epoch + 1}.")
                except Exception as e:
                    print(f"Error loading checkpoint {latest_checkpoint_path}: {e}. Starting new training.")
                    model = initial_model # Fallback to initial model if load fails
                    start_epoch = 0 # Reset start epoch
            else:
                print("No valid checkpoints found. Starting new training.")
                model = initial_model
        else:
            print("No checkpoints found. Starting new training.")
            model = initial_model
    else:
        print("config.force_train is True. Starting new training despite existing checkpoints.")
        model = initial_model 

    # Create datasets and dataloaders
    full_train_dataset = ImageTextDataset(cfg.train_dir, preprocess) 

    # --- Split dataset into training and validation sets ---
    dataset_size = len(full_train_dataset)
    validation_split = 0.15 
    indices = list(range(dataset_size))
    random.shuffle(indices) 
    
    split = int(validation_split * dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices) 

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Train the model
    print("\n--- Starting Model Training ---")

    final_trained_model, all_epoch_model_paths = train_clip_hybrid(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
        start_epoch=start_epoch 
    )
    print("\n--- Training Complete ---")

    # --- MODIFIED: Loop through all saved models for evaluation ---
    print("\n--- Starting Evaluation for All Trained Epochs ---")
    
    if not all_epoch_model_paths:
        print("No full model checkpoints were saved during training to evaluate.")
        # Fallback: if no epoch-wise models were saved, evaluate the final trained model
        print("Evaluating the final trained model from the last epoch.")
        final_epoch_for_results = cfg.epochs 
        if start_epoch > 0:
            final_epoch_for_results = start_epoch + (cfg.epochs - start_epoch)
        
        # Call evaluation for the final model
        try:
            perform_retrieval_and_save_results(
                model=final_trained_model, 
                query_dir=cfg.query_dir,
                gallery_dir=cfg.gallery_dir,
                top_k=cfg.top_k,
                distance_metric=cfg.distance_metric,
                device=device,
                batch_size=cfg.batch_size,
                epoch_number=final_epoch_for_results 
            )
        except Exception as e:
            print(f"Error during final model evaluation: {e}")
    else:
        # Sort the paths by epoch number to ensure ordered evaluation
        all_epoch_model_paths.sort(key=lambda path: int(re.search(r'model_epoch_(\d+)\.pt', os.path.basename(path)).group(1)))

        for model_path_for_eval in all_epoch_model_paths:
            # Extract epoch number from the path for the results file name
            match = re.search(r'model_epoch_(\d+)\.pt', os.path.basename(model_path_for_eval))
            if not match:
                print(f"Skipping evaluation for {model_path_for_eval}: could not extract epoch number.")
                continue

            current_eval_epoch = int(match.group(1))

            print(f"Evaluating model from: {model_path_for_eval} (Epoch {current_eval_epoch})")
            
            try:
                # Load the specific model for this epoch
                model_to_eval = torch.load(model_path_for_eval, map_location=device, weights_only=False) 
                model_to_eval.eval() # Set to eval mode

                perform_retrieval_and_save_results(
                    model=model_to_eval,
                    query_dir=cfg.query_dir,
                    gallery_dir=cfg.gallery_dir,
                    top_k=cfg.top_k,
                    distance_metric=cfg.distance_metric,
                    device=device,
                    batch_size=cfg.batch_size,
                    epoch_number=current_eval_epoch 
                )
            except Exception as e:
                print(f"Error during evaluation of epoch {current_eval_epoch} ({model_path_for_eval}): {e}")

    print("\n--- All Epochs Evaluation Complete ---")

if __name__ == "__main__":
    check_and_train_model()
