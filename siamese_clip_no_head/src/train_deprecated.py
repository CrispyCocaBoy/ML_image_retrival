import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from src.distance import calculate_distance
from src.test import fast_test, submit
import wandb

# == Loss function ==
def contrastive_loss(distance, labels, margin=1.0):
    positive = labels * distance.pow(2)
    negative = (1 - labels) * F.relu(margin - distance).pow(2)
    return (positive + negative).mean()

# == Training loop ==
def train_siamese(
    model,
    dataloader,
    device,
    epochs,
    margin,
    lr,
    momentum,
    weight_decay,
    optimizer_name,     # "SGD" or "adam"
    use_checkpoint,
    early_stops,
    query_loader,
    gallery_loader,
    patience,
    scheduler_name= None,  # "plateau" or "none"
    save_weights=True
):
    model.to(device)
    model.train()

    # === Init Weights & Biases (W&B) ===
    wandb.init(
        project="siamese-retrieval",
        entity="Simple_Guys",
        name=f"run_margin{margin}_lr{lr}",
        config={
            "architecture": "Siamese Network",
            "epochs": epochs,
            "margin": margin,
            "lr": lr,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "early_stops": early_stops
        }
    )


    # === Optimizer ===
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer must be 'SGD' or 'adam'")

    # === Scheduler ===
    scheduler = None
    if scheduler_name == "plateau" and early_stops == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # perché monitoriamo accuracy
            factor=0.5,  # dimezza il learning rate se no improvement
            patience=2,  # numero epoche senza miglioramento
            threshold=1e-4,
            cooldown=1,
            min_lr=1e-6
        )

    # === Checkpoint loading ===
    best_loss = float('inf')
    best_accuracy = 0.0
    start_epoch = 1
    no_improvement = 0

    if use_checkpoint:
        checkpoints = sorted(glob("repository/checkpoints/epoch_*.pth"))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from {latest_ckpt}, resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    epochs = start_epoch + epochs

    # === Initial accuracy evaluation ===
    if save_weights and early_stops:
        print("Evaluating model before training...")
        retrievals = fast_test(model, query_loader, gallery_loader, device, k=5)
        best_accuracy = submit(retrievals)
        print(f"Initial best_accuracy = {best_accuracy:.4f}")
    else:
        current_acc = None

    # === Training loop ===
    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for img1, img2, labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            distance = calculate_distance(emb1, emb2)
            loss = contrastive_loss(distance, labels, margin=margin)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f} | 🔁 LR: {current_lr:.6f}")

        # === Save weights ===
        if save_weights:
            if avg_loss < best_loss and not early_stops:
                best_loss = avg_loss
                torch.save(model.state_dict(), "repository/best_model/model_no_acc.pt")
                print("Saved best model (loss-based)")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"repository/checkpoints/epoch_{epoch}.pth")
            torch.save(model.state_dict(), f"repository/all_weights/model_epoch{epoch}.pt")

        # === Accuracy + Early Stopping ===
        if early_stops:
            retrievals = fast_test(model, query_loader, gallery_loader, device)
            current_acc = submit(retrievals)

            if current_acc > best_accuracy + 1e-4:
                best_accuracy = current_acc
                no_improvement = 0
                torch.save(model.state_dict(), "repository/best_model/model.pt")
                print(f"New best accuracy: {best_accuracy:.4f} — model saved.")
            else:
                no_improvement += 1
                print(f"No improvement: {no_improvement}/{patience} epochs")

            # Step the scheduler
            if scheduler:
                scheduler.step(current_acc)

            if no_improvement >= patience:
                print("Early stopping triggered")
                break

        elif scheduler:
            scheduler.step(avg_loss)  # fallback per schedulers non basati su accuracy

        # === Log to W&B ===
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": current_acc,
            "learning_rate": current_lr
        })

    print("Training completed.")

