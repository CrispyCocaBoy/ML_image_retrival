import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

def training_loop(train_loader,
                  model,
                  optimizer_type="adam",
                  epochs=10,
                  loss="triplet",
                  lr=1e-4,
                  margin=1.0,
                  weight_decay=0.0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # === OPTIMIZER ===
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # === LOSS FUNCTION ===
    if loss == "triplet":
        def triplet_loss(anchor, positive, negative, margin):
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
            return F.relu(pos_dist - neg_dist + margin).mean()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

    # === TRAINING LOOP ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for anchors, positives, negatives in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            loss_val = triplet_loss(anchor_emb, positive_emb, negative_emb, margin)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
