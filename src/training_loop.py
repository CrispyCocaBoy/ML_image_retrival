import os
import torch
from config import config
from src.finetuned_clip import FineTunedCLIP
from src.triplet_dataset import TripletDataset
from src.train_triplet import train
from torch.utils.data import DataLoader
from torchvision import transforms

def check_and_train():
    model_path = "finetuned_clip.pth"
    device = config.device

    if os.path.exists(model_path):
        print("✅ Modello fine-tuned già esistente. Salto training.")
        return

    print("🚀 Avvio training del modello fine-tuned CLIP...")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    train_dataset = TripletDataset(config.train_dir, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)
    train(model, train_loader, device, epochs=config.epochs, lr=config.learning_rate)

    torch.save(model.state_dict(), model_path)
    print(f"✅ Modello addestrato salvato in {model_path}")