import torch
import pandas as pd

@torch.no_grad()
def extract_embeddings(dataloader, model, device):
    model.eval()
    results = []

    for images, names in dataloader:
        images = images.to(device)
        embs = model(images).cpu() # qua è dove effettivamente facciamo girare i dati nel nostro modello

        for name, emb in zip(names, embs):
            results.append({
                "filename": name,
                "embedding": emb.numpy()  # convertiamo in array per uso successivo
            })

    # Converti in DataFrame
    df = pd.DataFrame(results)
    return df

