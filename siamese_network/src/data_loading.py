# == Package ==
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# == TrainingDataset (senza cv2 e senza augmentation) ==
class TrainingDataset(Dataset):
    def __init__(self, root_dir, transform, seed, num_pairs) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.seed = seed
        self.num_pairs = num_pairs

        # Mappa: classe -> lista immagini
        self.class_to_images = {}
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                imgs = [
                    img for img in os.listdir(class_path)
                    if img.lower().endswith(('jpg', 'jpeg', 'png'))
                ]
                if len(imgs) >= 2:
                    self.class_to_images[class_name] = imgs

        self.classes = list(self.class_to_images.keys())

        # 50% positive / 50% negative
        half = num_pairs // 2
        self.labels = [1]*half + [0]*(num_pairs-half)
        rnd = random.Random(seed)
        rnd.shuffle(self.labels)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx)
        label = self.labels[idx]

        if label == 1:
            # coppia positiva
            cls = rng.choice(self.classes)
            img1, img2 = rng.sample(self.class_to_images[cls], 2)
            path1 = os.path.join(self.root_dir, cls, img1)
            path2 = os.path.join(self.root_dir, cls, img2)
        else:
            # coppia negativa
            c1, c2 = rng.sample(self.classes, 2)
            img1 = rng.choice(self.class_to_images[c1])
            img2 = rng.choice(self.class_to_images[c2])
            path1 = os.path.join(self.root_dir, c1, img1)
            path2 = os.path.join(self.root_dir, c2, img2)

        # carica immagini
        im1 = Image.open(path1).convert("RGB")
        im2 = Image.open(path2).convert("RGB")

        # applica solo transform standard
        im1 = self.transform(im1)
        im2 = self.transform(im2)

        return im1, im2, label


# == QueryDataset (invariato) ==
class QueryDataset(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('jpg','jpeg','png')):
                    self.image_paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        im = Image.open(p).convert("RGB")
        im = self.transform(im)
        idr = os.path.relpath(p, self.transform.__dict__.get('root_dir', ''))
        return im, idr


# == GalleryDataset (invariato) ==
class GalleryDataset(QueryDataset):
    pass


# == Trasformazioni CLIP ==
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


# == Funzione per istanziare i loader ==
def datasets(
    training_dir: str,
    query_dir:    str,
    gallery_dir:  str,
    batch_size:   int,
    seed:         int,
    num_pairs:    int,
    num_val:      int,
    num_workers:  int,
    transform     = clip_transform,
):
    # --- train ---
    train_ds = TrainingDataset(
        root_dir=training_dir,
        transform= clip_transform,
        seed=seed,
        num_pairs=num_pairs
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    # --- validation (stesso dataset, ma senza shuffle) ---
    val_ds = TrainingDataset(
        root_dir=training_dir,
        transform=clip_transform,
        seed=seed+1,      # seed diversa così campiona coppie differenti
        num_pairs=num_val
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )

    # --- query & gallery ---
    query_ds = QueryDataset(root_dir=query_dir, transform=clip_transform)
    gallery_ds = GalleryDataset(root_dir=gallery_dir, transform=clip_transform)

    query_loader   = DataLoader(query_ds,   batch_size=1,            shuffle=False, num_workers=num_workers)
    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size,    shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader, query_loader, gallery_loader
