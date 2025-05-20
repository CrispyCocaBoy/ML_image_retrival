def training_loop(train_loader,
                  model,
                  optimizer_type,
                  epochs=10,
                  loss="triplet"):

    import torch.nn.functional as F
    import torch
    from tqdm import tqdm
    from torch import optim

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=1e-4)

    if loss == "triplet":
        def triplet_loss(anchor, positive, negative, margin=1.0):
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
            return F.relu(pos_dist - neg_dist + margin).mean()


        for epoch in range(epochs):
            model.train() # mette il modello in modalitò allenamento
            running_loss = 0.0
            for anchors, positives, negatives in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                anchors = anchors.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)

                anchor_emb = model(anchors)
                positive_emb = model(positives)
                negative_emb = model(negatives)

                loss_val = triplet_loss(anchor_emb, positive_emb, negative_emb)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                running_loss += loss_val.item()

            print(f"Epoch {epoch+1} - Avg Loss: {running_loss / len(train_loader):.4f}")
