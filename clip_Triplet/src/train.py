import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import os
import json
import shutil
from torch.utils.data import DataLoader
from clip_Triplet.config import config
from clip_Triplet.src.triplet_dataset import TripletDataset
from clip_Triplet.src.results import load_images_from_folder, get_top_k
from clip_Triplet.src.extract_embeddings_clip import extract_clip_embeddings
from torchvision import transforms

# Safe project root handling
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_DIR = os.path.join(BASE_DIR, "repository")
LOGS_DIR = os.path.join(REPO_DIR, "logs")
RETRIEVAL_DIR = os.path.join(REPO_DIR, "retrieval_repository")

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

def get_triplet_loader(root_dir, allowed_classes, transform, batch_size):
    dataset = TripletDataset(
        root_dir=root_dir,
        transform=transform,
        allowed_classes=allowed_classes
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            out_anchor = model(anchor)
            out_positive = model(positive)
            out_negative = model(negative)

            loss = criterion(out_anchor, out_positive, out_negative)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def prepare_data_for_retrieval():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    query_paths, query_images = load_images_from_folder(config.query_dir, transform=test_transform)
    gallery_paths, gallery_images = load_images_from_folder(config.gallery_dir, transform=test_transform)

    return query_paths, query_images, gallery_paths, gallery_images

@torch.no_grad()
def generate_results(model, device, epoch_label, query_paths, query_images, gallery_paths, gallery_images):
    query_features = extract_clip_embeddings(model, query_images, device)
    gallery_features = extract_clip_embeddings(model, gallery_images, device)

    results = get_top_k(query_features, gallery_features, gallery_paths, query_paths,
                        k=config.top_k, distance=config.distance_metric)

    os.makedirs(RETRIEVAL_DIR, exist_ok=True)
    json_path = os.path.join(RETRIEVAL_DIR, f"retrieval_results_{epoch_label}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {json_path}")

def train(model, train_loader, val_loader, device="cuda", epochs=10, lr=1e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    criterion = TripletLoss(margin=config.margin)

    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, "triplet_training.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if config.compute_validation_loss:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        else:
            writer.writerow(["epoch", "train_loss"])

    if config.save_retrieval_results_per_epoch:
        if os.path.exists(RETRIEVAL_DIR):
            shutil.rmtree(RETRIEVAL_DIR)
        os.makedirs(RETRIEVAL_DIR, exist_ok=True)

    query_paths, query_images, gallery_paths, gallery_images = prepare_data_for_retrieval()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for anchor, positive, negative in loop:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            out_anchor = model(anchor)
            out_positive = model(positive)
            out_negative = model(negative)

            loss = criterion(out_anchor, out_positive, out_negative)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        if config.compute_validation_loss:
            avg_val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = None
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if avg_val_loss is not None:
                writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
            else:
                writer.writerow([epoch+1, avg_train_loss])

        if config.save_retrieval_results_per_epoch:
            generate_results(model, device, epoch+1, query_paths, query_images, gallery_paths, gallery_images)

    generate_results(model, device, "final", query_paths, query_images, gallery_paths, gallery_images)

    return model
