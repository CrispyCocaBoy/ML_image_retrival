import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=2)

def pairwise_distance(embeddings):
    # Computes pairwise L2 distances
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=0.0)
    return distances

def get_semi_hard_triplets(embeddings, labels, margin):
    triplets = []
    distance_matrix = pairwise_distance(embeddings)
    
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        anchor = embeddings[i]

        # Get positive indices
        positive_indices = torch.where(labels == anchor_label)[0]
        positive_indices = positive_indices[positive_indices != i]

        # Get negative indices
        negative_indices = torch.where(labels != anchor_label)[0]

        for pos_idx in positive_indices:
            ap_dist = distance_matrix[i, pos_idx]

            # Find negatives that are farther than the positive but within margin
            semi_hard_negatives = [neg_idx for neg_idx in negative_indices 
                                   if ap_dist < distance_matrix[i, neg_idx] < ap_dist + margin]

            if semi_hard_negatives:
                neg_idx = semi_hard_negatives[0]  # pick one
                triplets.append((i, pos_idx.item(), neg_idx))
    return triplets

def train(model, dataloader, device="cuda", epochs=10, lr=1e-4, margin=1.0):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TripletLoss(margin=margin)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            triplets = get_semi_hard_triplets(embeddings, labels, margin)
            if not triplets:
                continue

            anchors = torch.stack([embeddings[a] for a, p, n in triplets])
            positives = torch.stack([embeddings[p] for a, p, n in triplets])
            negatives = torch.stack([embeddings[n] for a, p, n in triplets])

            loss = criterion(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model
