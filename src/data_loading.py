# Dataset per siamese network
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import os
import random
from tqdm import tqdm

cache_images = False

# Dataset per query/gallery
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform, target_transform=None, cache_images=cache_images):
        self.image_paths = list(Path(folder_path).glob("*.jpg"))
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        if cache_images:
            print(f"⏳ Caching {len(self.image_paths)} immagini in memoria...")
            for path in tqdm(self.image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    self.image_cache[path] = self.transform(img)
                except Exception as e:
                    print(f"Errore caricando {path}: {e}")
            print("✅ Caching immagini completato.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if self.cache_images and path in self.image_cache:
            img = self.image_cache[path]
        else:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            if self.cache_images:
                self.image_cache[path] = img
        return img, path.name

# Dataset per il training di Siamese Network
class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None, random_aug=False, cache_images=cache_images):
        self.root_dir = root_dir
        self.transform = transform
        self.random_aug = random_aug
        self.random_aug_prob = 0.7
        self.cache_images = cache_images
        self.image_cache = {}
        
        # Organizza le immagini per classe (cartella)
        self.class_to_images = {}
        self.data = []
        
        print("⏳ Organizzazione dataset...")
        # Itera attraverso tutte le cartelle nel root_dir
        for class_folder in tqdm(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                # Raccogli tutte le immagini in questa cartella
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) >= 2:  # Assicurati che ci siano almeno 2 immagini per classe
                    self.class_to_images[class_folder] = images
                    # Aggiungi tutte le possibili coppie di immagini della stessa classe
                    for i in range(len(images)):
                        for j in range(i + 1, len(images)):
                            self.data.append((
                                os.path.join(class_folder, images[i]),
                                os.path.join(class_folder, images[j]),
                                1  # Label 1 per immagini della stessa classe
                            ))
        
        # Aggiungi coppie negative (immagini di classi diverse)
        classes = list(self.class_to_images.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class1 = classes[i]
                class2 = classes[j]
                # Prendi un'immagine casuale da ogni classe
                img1 = random.choice(self.class_to_images[class1])
                img2 = random.choice(self.class_to_images[class2])
                self.data.append((
                    os.path.join(class1, img1),
                    os.path.join(class2, img2),
                    0  # Label 0 per immagini di classi diverse
                ))
        
        if cache_images:
            print("⏳ Caching immagini in memoria...")
            for img1_path, img2_path, _ in tqdm(self.data):
                for path in [img1_path, img2_path]:
                    full_path = os.path.join(root_dir, path)
                    if full_path not in self.image_cache:
                        try:
                            img = Image.open(full_path).convert("RGB")
                            self.image_cache[full_path] = self.transform(img)
                        except Exception as e:
                            print(f"Errore caricando {full_path}: {e}")
            print("✅ Caching immagini completato.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.data[idx]
        
        # Carica le immagini
        full_path1 = os.path.join(self.root_dir, img1_path)
        full_path2 = os.path.join(self.root_dir, img2_path)
        
        if self.cache_images:
            img1 = self.image_cache[full_path1]
            img2 = self.image_cache[full_path2]
        else:
            img1 = Image.open(full_path1).convert("RGB")
            img2 = Image.open(full_path2).convert("RGB")
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
        return img1, img2, label

    def random_augmentation(self, img, prob):
        if random.random() < prob:
            # Implementa qui le tue augmentation
            pass
        return img

# Funzione di caricamento generale
def retrival_data_loading(train_data_root, query_data_root, gallery_data_root, batch_size=32, num_workers=4, prefetch_factor=2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset training (base ImageFolder, non triplet)
    train_dataset = SiameseDataset(train_data_root, transform=transform, cache_images=cache_images)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    # Dataset query & gallery
    query_dataset = TestImageDataset(query_data_root, transform, cache_images=cache_images)
    gallery_dataset = TestImageDataset(gallery_data_root, transform, cache_images=cache_images)

    query_loader = DataLoader(
        query_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    return train_loader, query_loader, gallery_loader