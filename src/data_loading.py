from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random

# Dataset triplet per contrastive learning
class TripletDataset(Dataset):
    def __init__(self, image_folder_dataset):
        self.transform = image_folder_dataset.transform
        self.class_to_paths = defaultdict(list)
        self.data = []

        for path, class_idx in image_folder_dataset.imgs:
            class_name = image_folder_dataset.classes[class_idx]
            self.class_to_paths[class_name].append(path)
            self.data.append((class_name, path))

    def __getitem__(self, index):
        cls, anchor_path = self.data[index]
        positive_path = random.choice([p for p in self.class_to_paths[cls] if p != anchor_path])
        negative_cls = random.choice([c for c in self.class_to_paths if c != cls])
        negative_path = random.choice(self.class_to_paths[negative_cls])

        def load_img(p): return self.transform(Image.open(p).convert("RGB"))

        return load_img(anchor_path), load_img(positive_path), load_img(negative_path)

    def __len__(self):
        return len(self.data)

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
def retrival_data_loading(train_data_root, query_data_root, gallery_data_root, triplet_loss=False, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset training
    train_dataset = datasets.ImageFolder(root=train_data_root, transform=transform)

    if triplet_loss:
        train_dataset = TripletDataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Dataset query & gallery
    query_dataset = TestImageDataset(query_data_root, transform)
    gallery_dataset = TestImageDataset(gallery_data_root, transform)

    query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, query_loader, gallery_loader

