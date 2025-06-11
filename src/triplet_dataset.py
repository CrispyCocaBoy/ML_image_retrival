import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_images = self._load_images_by_class()

        self.valid_classes = [cls for cls, imgs in self.class_to_images.items() if len(imgs) >= 2]
        if not self.valid_classes:
            raise ValueError("⚠️ Nessuna classe con almeno 2 immagini trovata!")

    def _load_images_by_class(self):
        class_to_imgs = {}
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                images = [
                    os.path.join(class_path, img)
                    for img in os.listdir(class_path)
                    if img.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if images:
                    class_to_imgs[class_name] = images
        return class_to_imgs

    def __len__(self):
        return len(self.valid_classes)

    def __getitem__(self, idx):
        class_names = self.valid_classes
        anchor_class = class_names[idx % len(class_names)]
        positive_imgs = self.class_to_images[anchor_class]

        anchor_path, positive_path = random.sample(positive_imgs, 2)

        negative_class = random.choice([cls for cls in class_names if cls != anchor_class])
        negative_imgs = self.class_to_images[negative_class]
        negative_path = random.choice(negative_imgs)

        anchor = self.transform(Image.open(anchor_path).convert("RGB"))
        positive = self.transform(Image.open(positive_path).convert("RGB"))
        negative = self.transform(Image.open(negative_path).convert("RGB"))

        return anchor, positive, negative