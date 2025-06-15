import re
import os
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob


def contrastive_loss(distance, labels, margin=1.0):
    positive = labels * distance.pow(2)
    negative = (1 - labels) * F.relu(margin - distance).pow(2)
    return (positive + negative).mean()

def train_loop(
        model,
        dataloader_train,
        dataloader_validation,
        device,
        epochs,
        lr,
        momentum,
        weight_decay,
        optimizer_name,
        use_checkpoint,
        early_stops,
        patience,
        save_weights=True,
        margin=1.0,
        gradient_accumulation_steps=1
):
    model.to(device)
    model.train()

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer must be 'SGD', 'Adam', or 'AdamW'")

    best_val_loss = float('inf')
    no_improvement = 0
    start_epoch = 1

    def _epoch_num(path):
        m = re.search(r"epoch_(\d+)\.pth$", path)
        return int(m.group(1)) if m else -1

    if use_checkpoint:
        checkpoints = glob("repository/checkpoints/epoch_*.pth")
        if checkpoints:
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

    total_epochs = epochs if start_epoch == 1 else (start_epoch + epochs - 1)

    def evaluate_validation_loss():
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, labels in dataloader_validation:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
                z1, z2 = model(x1, x2)
                distance = F.pairwise_distance(z1, z2)
                loss = contrastive_loss(distance, labels, margin)
                val_loss += loss.item()
        return val_loss / len(dataloader_validation)

    if save_weights and early_stops:
        print("Evaluating model before training...")
        best_val_loss = evaluate_validation_loss()
        print(f"Initial best_val_loss = {best_val_loss:.4f}")

    log_data = []

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch}/{total_epochs}")

        for i, (x1, x2, labels) in enumerate(progress_bar):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
            z1, z2 = model(x1, x2)
            distance = F.pairwise_distance(z1, z2)
            loss = contrastive_loss(distance, labels, margin)
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader_train)
        val_loss = evaluate_validation_loss()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | 🔁 LR: {current_lr:.6f}")

        log_data.append({'epoch': epoch, 'avg_train_loss': avg_loss, 'avg_val_loss': val_loss})

        if save_weights:
            os.makedirs("repository/checkpoints", exist_ok=True)
            os.makedirs("repository/all_weights", exist_ok=True)
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
            torch.save(ckpt, f"repository/checkpoints/epoch_{epoch}.pth")
            torch.save(model.state_dict(), f"repository/all_weights/model_epoch_{epoch}.pt")

        if early_stops:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improvement = 0
                os.makedirs("repository/best_model", exist_ok=True)
                torch.save(model.state_dict(), "repository/best_model/model.pt")
                print(f"New best validation loss: {best_val_loss:.4f} — model saved.")
            else:
                no_improvement += 1
                print(f"No improvement: {no_improvement}/{patience} epochs")
            if no_improvement >= patience:
                print("Early stopping triggered")
                break

    csv_path = "repository/train_logs/loss_log.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['epoch', 'avg_train_loss', 'avg_val_loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_data:
            writer.writerow(row)

    print("Training completed. Loss log saved to:", csv_path)
