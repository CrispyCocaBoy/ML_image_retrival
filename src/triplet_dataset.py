import torch
from torch.utils.data import Dataset
from torch.nn.functional import pairwise_distance
from collections import defaultdict
from PIL import Image
import random
from tqdm import tqdm

class TripletDataset(Dataset):
    def __init__(self, image_folder_dataset, mining_strategy="random", model=None, device="cpu", margin=0.2, cache_images=True):
        self.transform = image_folder_dataset.transform
        self.device = device
        self.margin = margin
        self.cache_images = cache_images
        self.mining_strategy = mining_strategy
        self.model = model.eval().to(device) if model else None
        self.class_to_paths = defaultdict(list)
        self.data = []
        self.image_cache = {}
        self.embedding_cache = {}

        for path, class_idx in image_folder_dataset.imgs:
            class_name = image_folder_dataset.classes[class_idx]
            self.class_to_paths[class_name].append(path)
            self.data.append((class_name, path))

        if self.cache_images:
            print("⏳ Caching immagini in memoria...")
            for _, path in self.data:
                try:
                    img = Image.open(path).convert("RGB")
                    self.image_cache[path] = self.transform(img)
                except Exception as e:
                    print(f"Errore caricando {path}: {e}")
            print("✅ Caching immagini completato.")

        if self.model and self.mining_strategy == "semi-hard":
            self.refresh_embeddings(self.model)

    def load_img(self, path):
        if self.cache_images and path in self.image_cache:
            return self.image_cache[path]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def refresh_embeddings(self, model):
        """Aggiorna la cache degli embedding in base al modello attuale"""
        self.embedding_cache.clear()
        model.eval()
        with torch.no_grad():
            for _, path in tqdm(self.data, desc="🔄 Refresh embeddings"):
                img_tensor = self.load_img(path).unsqueeze(0).to(self.device)
                emb = model(img_tensor).squeeze(0).cpu()
                self.embedding_cache[path] = emb

    def __getitem__(self, index):
        cls, anchor_path = self.data[index]
        anchor_img = self.load_img(anchor_path)

        if self.mining_strategy == "random":
            positive_path = random.choice([p for p in self.class_to_paths[cls] if p != anchor_path])
            negative_cls = random.choice([c for c in self.class_to_paths if c != cls])
            negative_path = random.choice(self.class_to_paths[negative_cls])

        elif self.mining_strategy == "semi-hard":
            anchor_emb = self.embedding_cache[anchor_path]
            positive_path = random.choice([p for p in self.class_to_paths[cls] if p != anchor_path])
            positive_emb = self.embedding_cache[positive_path]
            pos_dist = pairwise_distance(anchor_emb.unsqueeze(0), positive_emb.unsqueeze(0), p=2).item()

            negative_path = None
            for neg_cls in random.sample([c for c in self.class_to_paths if c != cls], k=min(10, len(self.class_to_paths) - 1)):
                for neg_path in random.sample(self.class_to_paths[neg_cls], k=min(10, len(self.class_to_paths[neg_cls]))):
                    neg_emb = self.embedding_cache[neg_path]
                    neg_dist = pairwise_distance(anchor_emb.unsqueeze(0), neg_emb.unsqueeze(0), p=2).item()
                    if pos_dist < neg_dist < pos_dist + self.margin:
                        negative_path = neg_path
                        break
                if negative_path:
                    break

            if not negative_path:
                negative_cls = random.choice([c for c in self.class_to_paths if c != cls])
                negative_path = random.choice(self.class_to_paths[negative_cls])

        else:
            raise ValueError(f"Strategia non riconosciuta: {self.mining_strategy}")

        positive_img = self.load_img(positive_path)
        negative_img = self.load_img(negative_path)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.data)
