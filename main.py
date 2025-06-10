import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from config import config
from src.finetuned_clip import FineTunedCLIP
from src.folder_dataset import FolderDataset
from src.train_triplet import train
from src.extract_embeddings_generic import extract_embeddings
from src.efficientnet_embedder import EfficientNetEmbedder
from torchvision.models import EfficientNet_B0_Weights
from src.results import get_top_k
from src.folder_dataset import FolderDataset, InferenceFolderDataset, FlatInferenceDataset

def main():
    device = config.device

    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=(0.48145466, 0.4578275, 0.40821073),
    #         std=(0.26862954, 0.26130258, 0.27577711)
    #     )
    # ])

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # === Dataset di training (con triplette) ===
    train_dataset = FolderDataset(config.train_dir, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # === Model ===
    #model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)
    model = EfficientNetEmbedder(embed_dim=config.embedding_dim, freeze_backbone=False)


    #model_path = "finetuned_clip.pth"
    model_path = "efficientnet_triplet.pth"
    if config.force_train or not os.path.exists(model_path):
        print("🚀 Avvio training del modello fine-tuned efficient_net...")
        train(model, train_loader, device, epochs=config.epochs, lr=config.learning_rate)
        torch.save(model.state_dict(), model_path)
        print("✅ Modello fine-tuned salvato come", model_path)
    else:
        print("✅ Modello fine-tuned già esistente. Salto training.")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # === Test datasets using FolderDataset + DataLoader ===
    query_dataset = FlatInferenceDataset(config.query_dir, transform=preprocess)
    query_loader = DataLoader(query_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    gallery_dataset = FlatInferenceDataset(config.gallery_dir, transform=preprocess)
    gallery_loader = DataLoader(gallery_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # query_features = extract_clip_embeddings(model, query_images, device)
    # gallery_features = extract_clip_embeddings(model, gallery_images, device)

    query_features, query_paths = extract_embeddings(model, query_loader, device)
    gallery_features, gallery_paths = extract_embeddings(model, gallery_loader, device)

    results = get_top_k(query_features, gallery_features, gallery_paths, query_paths,
                        k=config.top_k, distance=config.distance_metric)

    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Ranking salvato in retrieval_results.json")

if __name__ == "__main__":
    main()
