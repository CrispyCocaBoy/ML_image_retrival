import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -----------------------------
# Dataset che legge immagini con label da cartelle
# -----------------------------
class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for root, dirs, files in os.walk(folder_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png')):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path))
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label, path

# -----------------------------
# Dataset che genera triplette (anchor, positive, negative)
# -----------------------------
class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = self._build_index()

    def _build_index(self):
        label_to_indices = {}
        for idx, (_, label, _) in enumerate(self.base_dataset):
            label_to_indices.setdefault(label, []).append(idx)
        return label_to_indices

    def __getitem__(self, index):
        anchor_img, anchor_label, _ = self.base_dataset[index]

        # Positive sample (stessa classe)
        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img, _, _ = self.base_dataset[positive_idx]

        # Negative sample (classe diversa)
        negative_label = random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img, _, _ = self.base_dataset[negative_idx]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.base_dataset)

# -----------------------------
# Rete neurale che genera embeddings
# -----------------------------
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # rimuove classificatore
        self.fc = nn.Linear(512, 128)  # embedding finale

    def forward(self, x):
        x = self.backbone(x)   # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # normalizza l'embedding

# -----------------------------
# Setup: device, data, modello, training loop
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10

# Dataset e DataLoader
base_dataset = CustomImageDataset("./data_example/training")
triplet_dataset = TripletDataset(base_dataset)
dataloader = DataLoader(triplet_dataset, batch_size=32, shuffle=True, num_workers=4)

# Modello + ottimizzatore + loss
model = EmbeddingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# -----------------------------
# Training
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
