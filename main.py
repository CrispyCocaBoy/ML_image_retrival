import os
import json
import torch
from torchvision import transforms
from src.build_model_clip import load_clip_model
from src.extract_embeddings_clip import extract_clip_embeddings
from src.results import load_images_from_folder, get_top_k
from config import config

def main():
    device = config.device

    # 1. Carica modello CLIP e preprocessing
    clip_model, clip_preprocess = load_clip_model(device)

    # 2. Carica immagini da query e gallery
    gallery_paths, gallery_imgs = load_images_from_folder(config.gallery_dir, clip_preprocess)
    query_paths, query_imgs = load_images_from_folder(config.query_dir, clip_preprocess)

    # 3. Estrai gli embedding
    gallery_embeddings = extract_clip_embeddings(clip_model, gallery_imgs, device, batch_size=config.batch_size)
    query_embeddings = extract_clip_embeddings(clip_model, query_imgs, device, batch_size=config.batch_size)

    # 4. Calcola le top-k immagini simili
    results = get_top_k(
        query_embeds=query_embeddings,
        gallery_embeds=gallery_embeddings,
        gallery_paths=gallery_paths,
        query_paths=query_paths,
        k=10
    )

    # 5. Salva i risultati
    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ retrieval_results.json salvato correttamente.")

if __name__ == "__main__":
    main()
