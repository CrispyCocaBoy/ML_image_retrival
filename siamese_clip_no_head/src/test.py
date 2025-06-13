# == Package ==
import json
import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image

# == Script ==
def fast_test(model, query_loader, gallery_loader, device, k=10):
    model.eval()
    model.to(device)

    # === Estrai tutti gli embeddings della gallery ===
    gallery_embeddings = []
    gallery_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(gallery_loader, desc="Extracting gallery embeddings"):
            images = images.to(device)
            emb = model(images)  # [B, 64]
            gallery_embeddings.append(emb)
            gallery_filenames.extend(filenames)

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)  # [N_gallery, 64]

    # === Estrai embeddings della query e confronta con tutta la gallery ===
    results = {}
    with torch.no_grad():
        for images, filenames in tqdm(query_loader, desc="Processing queries"):
            images = images.to(device)
            query_embeddings = model(images) # [B, 64]

            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            gallery_embeddings_norm = F.normalize(gallery_embeddings, p=2, dim=1)
            distances = 1 - torch.matmul(query_embeddings, gallery_embeddings_norm.T)

            # Prendi i top-k (minori distanze)
            topk = torch.topk(distances, k=k, dim=1, largest=False)

            for i, fname in enumerate(filenames):
                indices = topk.indices[i].tolist()
                retrieved_files = [gallery_filenames[idx] for idx in indices]
                results[fname] = retrieved_files

    return results


def submit(results, groupname = "Simple Guys", url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
        return result['accuracy']
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


def show_retrieved_images(results: dict, query_dir: str, gallery_dir: str, k: int = 5, n_queries: int = 5):
    """
    Visualizza i risultati di image retrieval.
    
    Args:
        results: dizionario query -> lista di immagini gallery
        query_dir: cartella con le immagini query
        gallery_dir: cartella con le immagini gallery
        k: numero di immagini recuperate da mostrare per ogni query
        n_queries: quante query mostrare
    """
    query_names = list(results.keys())[:n_queries]
    
    for query_name in query_names:
        retrieved_names = results[query_name][:k]

        # Crea nuova figura
        plt.figure(figsize=(15, 3))
        
        # Query image
        query_path = os.path.join(query_dir, query_name)
        query_img = Image.open(query_path)
        plt.subplot(1, k + 1, 1)
        plt.imshow(query_img)
        plt.title("Query")
        plt.axis("off")
        
        # Retrieved images
        for i, retrieved_name in enumerate(retrieved_names):
            gallery_path = os.path.join(gallery_dir, retrieved_name)
            gallery_img = Image.open(gallery_path)
            plt.subplot(1, k + 1, i + 2)
            plt.imshow(gallery_img)
            plt.title(f"Top {i+1}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()

def save_results_to_json(results: Dict[str, List[str]], output_path: str) -> None:
    """
    Salva i risultati del retrieval in un file JSON.

    Args:
        results: dizionario con chiavi i nomi delle immagini query e valori le liste di file retrieved.
        output_path: percorso completo del file JSON da salvare.
    """
    # Conversione in formato compatibile con JSON (assicurandosi che tutti i valori siano stringhe)
    serializable_results = {
        str(query): [str(fname) for fname in retrieved_list]
        for query, retrieved_list in results.items()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Risultati salvati in {output_path}")
