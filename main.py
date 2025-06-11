import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from config import config
from src.finetuned_clip import FineTunedCLIP
from src.triplet_dataset import TripletDataset
from src.train_triplet import train
from src.extract_embeddings_clip import extract_clip_embeddings
from src.results import get_top_k, load_images_from_folder

def main():
    device = config.device

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # === Dataset di training (con triplette) ===
    train_dataset = TripletDataset(config.train_dir, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # === Model ===
    model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)

    model_path = "finetuned_clip.pth"
    if config.force_train or not os.path.exists(model_path):
        print("🚀 Avvio training del modello fine-tuned CLIP...")
        train(model, train_loader, device, epochs=config.epochs, lr=config.learning_rate)
        torch.save(model.state_dict(), model_path)
        print("✅ Modello fine-tuned salvato come", model_path)
    else:
        print("✅ Modello fine-tuned già esistente. Salto training.")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # === Dataset di test (senza classi) ===
    query_paths, query_images = load_images_from_folder(config.query_dir, transform=preprocess)
    gallery_paths, gallery_images = load_images_from_folder(config.gallery_dir, transform=preprocess)

    query_features = extract_clip_embeddings(model, query_images, device)
    gallery_features = extract_clip_embeddings(model, gallery_images, device)

    results = get_top_k(query_features, gallery_features, gallery_paths, query_paths,
                        k=config.top_k, distance=config.distance_metric)

    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Ranking salvato in retrieval_results.json")

if __name__ == "__main__":
    main()