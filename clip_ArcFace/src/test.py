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


def fast_test(model, query_loader, gallery_loader, device, k=10):
    model.eval()
    model.to(device)

    gallery_embeddings = []
    gallery_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(gallery_loader, desc="Extracting gallery embeddings"):
            images = images.to(device)
            emb = model(images, return_logits=False)  # [B, 64]
            gallery_embeddings.append(emb)
            gallery_filenames.extend([os.path.basename(f) for f in filenames])

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)

    results = {}
    with torch.no_grad():
        for images, filenames in tqdm(query_loader, desc="Processing queries"):
            images = images.to(device)
            query_embeddings = model(images, return_logits=False)

            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            gallery_embeddings_norm = F.normalize(gallery_embeddings, p=2, dim=1)
            distances = 1 - torch.matmul(query_embeddings, gallery_embeddings_norm.T)

            topk = torch.topk(distances, k=k, dim=1, largest=False)

            for i, fname in enumerate(filenames):
                fname_base = os.path.basename(fname)
                indices = topk.indices[i].tolist()
                retrieved_files = [gallery_filenames[idx] for idx in indices]
                results[fname_base] = retrieved_files

    return results


# Versione conforme alla specifica richiesta
def submit(data: Dict[str, List[str]], group_name = "Simple_Guys", url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    """
    Invia i risultati del retrieval al server per la valutazione.

    Args:
        data: dizionario {query_image_name: [top_k_retrieved_images]}
        group_name: nome del gruppo per l’identificazione
        url: endpoint per la sottomissione (di default quello fornito)
    """
    payload = {
        "groupname": group_name,
        "images": data
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"accuracy is {result.get('accuracy', 'N/A')}")
        return result.get('accuracy', None)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print(f"Server response is not JSON:\n{response.text}")


def show_retrieved_images(results: dict, query_dir: str, gallery_dir: str, k: int = 5, n_queries: int = 5):
    query_names = list(results.keys())[:n_queries]

    for query_name in query_names:
        retrieved_names = results[query_name][:k]
        plt.figure(figsize=(15, 3))

        query_path = os.path.join(query_dir, query_name)
        query_img = Image.open(query_path)
        plt.subplot(1, k + 1, 1)
        plt.imshow(query_img)
        plt.title("Query")
        plt.axis("off")

        for i, retrieved_name in enumerate(retrieved_names):
            gallery_path = os.path.join(gallery_dir, retrieved_name)
            gallery_img = Image.open(gallery_path)
            plt.subplot(1, k + 1, i + 2)
            plt.imshow(gallery_img)
            plt.title(f"Top {i + 1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def save_results_to_json(results: Dict[str, List[str]], output_path: str) -> None:
    serializable_results = {
        str(query): [str(fname) for fname in retrieved_list]
        for query, retrieved_list in results.items()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Risultati salvati in {output_path}")
