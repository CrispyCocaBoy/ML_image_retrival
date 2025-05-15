import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# === SEED ===
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# === DATASET ===
class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        for root, _, files in os.walk(folder_path):
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

# === SPLIT ===
dataset = CustomImageDataset("./data_example/training")
query_indices = random.sample(range(len(dataset)), 5)
query_set = [dataset[i] for i in query_indices]
train_indices = [i for i in range(len(dataset)) if i not in query_indices]
train_subset = torch.utils.data.Subset(dataset, train_indices)

# === TRIPLET DATASET ===
class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.data = base_dataset
        self.label_to_indices = {}
        for idx in range(len(self.data)):
            _, label, _ = self.data[idx]
            self.label_to_indices.setdefault(label, []).append(idx)

    def __getitem__(self, index):
        anchor_img, anchor_label, _ = self.data[index]
        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img, _, _ = self.data[positive_idx]

        negative_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img, _, _ = self.data[negative_idx]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.data)

# === MODEL ===
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 53 * 53, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        return (self.embedding_net(anchor),
                self.embedding_net(positive),
                self.embedding_net(negative))

# === TRAINING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_net = EmbeddingNet().to(device)
model = TripletNet(embedding_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

triplet_dataset = TripletDataset(train_subset)
triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0
    for a, p, n in tqdm(triplet_loader):
        a, p, n = a.to(device), p.to(device), n.to(device)
        anchor_emb, pos_emb, neg_emb = model(a, p, n)
        loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss / len(triplet_loader):.4f}")

# === RETRIEVAL ===
embedding_net.eval()
train_loader = DataLoader(train_subset, batch_size=256)

gallery_embeddings, gallery_labels, gallery_images = [], [], []
with torch.no_grad():
    for imgs, labels, _ in train_loader:
        imgs = imgs.to(device)
        emb = embedding_net(imgs).cpu()
        gallery_embeddings.append(emb)
        gallery_labels.extend(labels)
        gallery_images.extend(imgs.cpu())

gallery_embeddings = torch.cat(gallery_embeddings, dim=0).numpy()

fig, axs = plt.subplots(5, 11, figsize=(18, 8))
for row, (q_img, q_label, _) in enumerate(query_set):
    q_tensor = q_img.unsqueeze(0).to(device)
    q_embedding = embedding_net(q_tensor).detach().cpu().numpy()
    sims = cosine_similarity(q_embedding, gallery_embeddings)[0]
    top10_idx = np.argsort(sims)[::-1][:10]

    axs[row, 0].imshow(q_img.permute(1, 2, 0).cpu().numpy())
    axs[row, 0].set_title(f"Query\n{q_label}")
    axs[row, 0].axis("off")

    for col, idx in enumerate(top10_idx):
        axs[row, col + 1].imshow(gallery_images[idx].permute(1, 2, 0).numpy())
        axs[row, col + 1].set_title(f"{gallery_labels[idx]}")
        axs[row, col + 1].axis("off")

plt.tight_layout()
plt.show()
