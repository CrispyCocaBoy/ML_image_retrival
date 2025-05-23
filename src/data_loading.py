# Dataset triplet per contrastive learning
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path

# Dataset per query/gallery
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.image_paths = list(Path(folder_path).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.image_paths[idx].name

# Funzione di caricamento generale

def retrival_data_loading(train_data_root, query_data_root, gallery_data_root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset training (base ImageFolder, non triplet)
    train_dataset = datasets.ImageFolder(root=train_data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Dataset query & gallery
    query_dataset = TestImageDataset(query_data_root, transform)
    gallery_dataset = TestImageDataset(gallery_data_root, transform)

    query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, query_loader, gallery_loader