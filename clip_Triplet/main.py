import os
import torch
from torchvision import transforms
from clip_Triplet.config import config
from clip_Triplet.src.model import FineTunedCLIP
from clip_Triplet.src.train import train, get_triplet_loader

def main():
    device = config.device

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    train_classes = [
        class_name for class_name in os.listdir(config.train_dir)
        if os.path.isdir(os.path.join(config.train_dir, class_name))
    ]

    val_classes = [
        class_name for class_name in os.listdir(config.validation_dir)
        if os.path.isdir(os.path.join(config.validation_dir, class_name))
    ]

    train_loader = get_triplet_loader(config.train_dir, train_classes, preprocess, config.batch_size)
    val_loader = get_triplet_loader(config.validation_dir, val_classes, preprocess, config.batch_size)

    model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)

    model_path = "finetuned_clip.pth"
    if config.force_train or not os.path.exists(model_path):
        print("Starting fine-tuning of CLIP model...")
        train(model, train_loader, val_loader, device, epochs=config.epochs, lr=config.learning_rate)
        torch.save(model.state_dict(), model_path)
        print("Fine-tuned model saved at", model_path)
    else:
        print("Fine-tuned model already exists. Skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))

if __name__ == "__main__":
    main()
