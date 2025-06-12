# main.py (excerpt of relevant imports and check_and_train)
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from config import config
from src.finetuned_clip import FineTunedCLIP
# Updated imports for new structure
from src.image_text_dataset import ImageTextDataset
from src.training_loop import train_clip_hybrid # Now only train_clip_hybrid and CLIPLoss are here
from src.extract_embeddings_clip import extract_clip_embeddings
from src.results import get_top_k, load_images_from_folder

def check_and_train():
    # This function should be in main.py
    model_path = "finetuned_clip.pth"
    device = config.device

    if config.force_train or not os.path.exists(model_path):
        print("🚀 Starting fine-tuning of CLIP model with hybrid loss...")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        train_dataset = ImageTextDataset(config.train_dir, transform=preprocess)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)
        train_clip_hybrid(model, train_loader, device, epochs=config.epochs, lr=config.learning_rate)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Trained model saved to {model_path}")
    else:
        print("✅ Fine-tuned model already exists. Skipping training.")

def main():
    device = config.device

    # Call the check_and_train function here
    check_and_train()

    # Load the model state after check_and_train completes
    model_path = "finetuned_clip.pth"
    model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ... rest of your main logic for retrieval ...
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    query_paths, query_images = load_images_from_folder(config.query_dir, transform=preprocess)
    gallery_paths, gallery_images = load_images_from_folder(config.gallery_dir, transform=preprocess)

    query_features = extract_clip_embeddings(model, query_images, device)
    gallery_features = extract_clip_embeddings(model, gallery_images, device)

    results = get_top_k(query_features, gallery_features, gallery_paths, query_paths,
                        k=config.top_k, distance=config.distance_metric)

    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Ranking saved in retrieval_results.json")

if __name__ == "__main__":
    main()