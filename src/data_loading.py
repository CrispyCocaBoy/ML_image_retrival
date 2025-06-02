# Dataset per siamese network
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import os

# Dataset per query/gallery
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform, target_transform = None):
        self.image_paths = list(Path(folder_path).glob("*.jpg"))
        self.transform = transform
        self.target_transform = target_transform

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

# Dataset per il training di Siamese Network
class SiameseDataset(Dataset):

    def __init__(self, root_dir, path_file_dir, transform=None, random_aug=False):
        self.root_dir = root_dir
        path_file = open(path_file_dir, 'r')
        data = []
        for line in path_file:
            line = line.strip()
            img1, img2, label = line.split(' ')
            label = int(label)
            data.append((img1, img2, label))
        self.data = data
        self.transform = transform
        self.random_aug = random_aug
        self.random_aug_prob = 0.7
        path_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        img1_file = Image.open(os.path.join(self.root_dir, img1))
        img2_file = Image.open(os.path.join(self.root_dir, img2))
        if self.random_aug:
            img1_file = self.random_augmentation(img1_file, self.random_aug_prob)
            img2_file = self.random_augmentation(img2_file, self.random_aug_prob)

        if self.transform:
            img1_file = self.transform(img1_file)
            img2_file = self.transform(img2_file)
        return (img1_file, img2_file, label)