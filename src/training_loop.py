import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
from pathlib import Path
from .loss import ContrastiveLoss

def train_siamese(
    model,
    train_loader,
    val_loader=None,
    optimizer_type="adam",
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=10,
    margin=2.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="model_repository",
    model_name="siamese_model"
):
    """
    Training loop for Siamese Network.
    
    Args:
        model: Siamese network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        optimizer_type: Type of optimizer ("adam" or "sgd")
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        epochs: Number of training epochs
        margin: Margin for contrastive loss
        device: Device to train on ("cuda" or "cpu")
        save_dir: Directory to save model checkpoints
        model_name: Base name for saved model files
    """
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss function
    criterion = ContrastiveLoss(margin=margin)
    
    # Initialize optimizer
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)     
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2,
    )
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for img1, img2, label in train_pbar:
            # Move data to device
            img1, img2 = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_steps += 1
            train_pbar.set_postfix({"loss": train_loss / train_steps})
        
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for img1, img2, label in val_pbar:
                    img1, img2 = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    
                    output1, output2 = model(img1, img2)
                    loss = criterion(output1, output2, label)
                    
                    val_loss += loss.item()
                    val_steps += 1
                    val_pbar.set_postfix({"loss": val_loss / val_steps})
            
            avg_val_loss = val_loss / val_steps
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, save_path / f"{model_name}_best.pth")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss if val_loader else None,
        }, save_path / f"{model_name}_epoch_{epoch+1}.pth")
        
        # Print epoch statistics
        if val_loader:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}")
    
    return model
