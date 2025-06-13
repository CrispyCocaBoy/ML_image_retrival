import re
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
import wandb
from src.loss import ArcFace


# == Training loop for classification with CrossEntropyLoss (Early Stopping on Validation Loss) ==
def train_loop(
        model,
        train_loader,
        validation_loader,
        device,
        epochs,
        lr,
        momentum,
        weight_decay,
        optimizer_name,
        use_checkpoint,
        early_stops,
        patience,
        save_weights=True
):
    model.to(device)
    model.train()

    wandb.init(
        project="clip-classification",
        entity="Simple_Guys",
        name=f"run_classification_lr{lr}",
        config={
            "architecture": "ViT + Classifier",
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer_name,
            "early_stops": early_stops
        }
    )

    # Select optimizer
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer must be 'SGD', 'Adam', or 'AdamW'")

    best_val_loss = float('inf')
    start_epoch = 1
    no_improvement = 0

    # Helper to extract epoch number
    def _epoch_num(path):
        m = re.search(r"epoch_(\d+)\.pth$", path)
        return int(m.group(1)) if m else -1

    # Load checkpoint if requested
    if use_checkpoint:
        checkpoints = glob("repository/checkpoints/epoch_*.pth")
        if checkpoints:
            # Sort by numeric epoch
            checkpoints = sorted(checkpoints, key=_epoch_num)
            latest_ckpt = checkpoints[-1]
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from {latest_ckpt}, resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

        for g in optimizer.param_groups:
            g['lr'] = lr

    # Adjust total epochs considering resumed training
    total_epochs = epochs if start_epoch == 1 else (start_epoch + epochs - 1)

    arcface = ArcFace(s=64.0, margin=0.5)
    criterion = nn.CrossEntropyLoss()

    def evaluate_validation_loss():
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                logits = arcface(logits, labels)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        return val_loss / len(validation_loader)

    # Initial validation
    if save_weights and early_stops:
        print("Evaluating model before training...")
        best_val_loss = evaluate_validation_loss()
        print(f"Initial best_val_loss = {best_val_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            logits = arcface(logits, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate_validation_loss()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | 🔁 LR: {current_lr:.6f}")

        # Save weights
        if save_weights:
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
            torch.save(ckpt, f"repository/checkpoints/epoch_{epoch}.pth")
            torch.save(model.state_dict(), f"repository/all_weights/model_epoch_{epoch}.pt")

        # Early stopping
        if early_stops:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improvement = 0
                torch.save(model.state_dict(), "repository/best_model/model.pt")
                print(f"New best validation loss: {best_val_loss:.4f} — model saved.")
            else:
                no_improvement += 1
                print(f"No improvement: {no_improvement}/{patience} epochs")

            if no_improvement >= patience:
                print("Early stopping triggered")
                break

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        })

    print("Training completed.")
