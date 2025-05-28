import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from src.embedding import extract_embeddings
from src.results import compute_results
from src.evaluation import evaluation



def training_loop(train_loader,
                  model,
                  epochs = 25,
                  query_loader=None,
                  gallery_loader=None,
                  ground_truth_path=None,
                  optimizer_type="adam",
                  loss="triplet",
                  lr=1e-4,
                  margin=1.0,
                  weight_decay=0.0,
                  early_stopping_patience=4,
                  model_save_path="best_model.pth",
                  top_k=3):

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

    # === EARLY STOPPING ===
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0

        pbar = tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{epochs}")
        for anchors, positives, negatives in pbar:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            with torch.no_grad():
                total_pos_dist += F.pairwise_distance(anchor_emb, positive_emb, p=2).mean().item()
                total_neg_dist += F.pairwise_distance(anchor_emb, negative_emb, p=2).mean().item()

            loss_val = triplet_loss(anchor_emb, positive_emb, negative_emb, margin)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()

        avg_loss = running_loss / len(train_loader)
        avg_pos = total_pos_dist / len(train_loader)
        avg_neg = total_neg_dist / len(train_loader)

        tqdm.write(f"✅ Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f} | D(ap): {avg_pos:.4f} | D(an): {avg_neg:.4f}")

        if hasattr(train_loader.dataset, "refresh_embeddings"):
            tqdm.write("🔄 Aggiornamento embeddings nel dataset...")
            train_loader.dataset.refresh_embeddings(model)

        # === EVALUATION ===
        model.eval()
        query_df = extract_embeddings(query_loader, model, device)
        gallery_df = extract_embeddings(gallery_loader, model, device)
        results = compute_results(query_df, gallery_df, metric="euclidean")

        accuracy = evaluation(results, ground_truth_path, top_k=top_k, verbose=False)
        tqdm.write(f"📉 Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            tqdm.write(f"✅ Nuovo best model salvato (accuracy {accuracy:.2f}%)")
        else:
            patience_counter += 1
            tqdm.write(f"⏳ Nessun miglioramento ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                tqdm.write("🚩 Early stopping triggered!")
                break

    print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%. Model saved to {model_save_path}.")
