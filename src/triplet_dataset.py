from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
import torch
import torch.nn.functional as F
import random
import time

class TripletDataset(Dataset):
    def __init__(self, image_folder_dataset, mining_strategy="random", model=None, device="cpu", margin=0.2, cache_images=True):
        self.transform = image_folder_dataset.transform
        self.class_to_paths = defaultdict(list)
        self.data = []
        self.mining_strategy = mining_strategy
        self.model = model.eval().to(device) if model else None
        self.device = device
        self.margin = margin
        self.cache_images = cache_images
        self.image_cache = {}  # Preload images here if needed

        for path, class_idx in image_folder_dataset.imgs:
            class_name = image_folder_dataset.classes[class_idx]
            self.class_to_paths[class_name].append(path)
            self.data.append((class_name, path))

        if self.cache_images:
            print("⏳ Caching images in memory...")
            for _, path in self.data:
                try:
                    self.image_cache[path] = self.transform(Image.open(path).convert("RGB"))
                except Exception as e:
                    print(f"Errore nel caching di {path}: {e}")
            print("✅ Caching completato.")

    def load_img(self, path):
        if self.cache_images and path in self.image_cache:
            return self.image_cache[path]
        return self.transform(Image.open(path).convert("RGB"))

    def get_embedding(self, img_tensor):
        with torch.no_grad():
            emb = self.model(img_tensor.unsqueeze(0).to(self.device))
            return emb.squeeze(0)

    def __getitem__(self, index):
        cls, anchor_path = self.data[index]
        anchor_img = self.load_img(anchor_path)

        if self.mining_strategy == "random":
            positive_path = random.choice([p for p in self.class_to_paths[cls] if p != anchor_path])
            negative_cls = random.choice([c for c in self.class_to_paths if c != cls])
            negative_path = random.choice(self.class_to_paths[negative_cls])

        elif self.mining_strategy == "semi-hard":
            if not self.model:
                raise ValueError("Model must be provided for semi-hard mining.")

            anchor_emb = self.get_embedding(anchor_img)
            positive_candidates = [p for p in self.class_to_paths[cls] if p != anchor_path]
            positive_path = random.choice(positive_candidates)
            positive_img = self.load_img(positive_path)
            positive_emb = self.get_embedding(positive_img)
            pos_dist = F.pairwise_distance(anchor_emb.unsqueeze(0), positive_emb.unsqueeze(0), p=2).item()

            negative_path = None
            for neg_cls in [c for c in self.class_to_paths if c != cls]:
                for neg_path in self.class_to_paths[neg_cls]:
                    neg_img = self.load_img(neg_path)
                    neg_emb = self.get_embedding(neg_img)
                    neg_dist = F.pairwise_distance(anchor_emb.unsqueeze(0), neg_emb.unsqueeze(0), p=2).item()

                    if pos_dist < neg_dist < pos_dist + self.margin:
                        negative_path = neg_path
                        break
                if negative_path:
                    break

            if not negative_path:
                negative_cls = random.choice([c for c in self.class_to_paths if c != cls])
                negative_path = random.choice(self.class_to_paths[negative_cls])

        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

        positive_img = self.load_img(positive_path)
        negative_img = self.load_img(negative_path)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.data)
