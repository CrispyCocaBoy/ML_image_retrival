import os
from torchvision.datasets import ImageFolder, VisionDataset, DatasetFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# Trasformazione compatibile con CLIP (ViT-L/14 e simili)
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

# Custom dataset
class gallery_query(Dataset):
    def __init__(self, image_dir, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(extensions)
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(path)
        return img, filename

def datasets(
    training_dir,
    query_dir,
    gallery_dir,
    validation_dir,
    batch_size=32,
    num_workers=4,
    transform=None,
    drop_last=True,
    verbose=True
):
    # Se non viene fornita una trasformazione, usa quella CLIP
    if transform is None:
        transform = clip_transform

    # Controllo esistenza directory
    for dir_path, name in zip([training_dir, query_dir, gallery_dir], ["training", "query", "gallery"]):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory '{name}' non trovata: {dir_path}")

    if verbose:
        print(f"Caricamento dati da:\n - Training: {training_dir}\n - Query: {query_dir}\n - Gallery: {gallery_dir}")

    # Caricamento dataset
    train_dataset = ImageFolder(root=training_dir, transform=transform)
    query_dataset = gallery_query(image_dir=query_dir, transform=transform)
    gallery_dataset = gallery_query(image_dir=gallery_dir, transform=transform)
    validation_dataset = ImageFolder(root=validation_dir, transform=transform)

    # Creazione dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=drop_last
    )
    query_loader = DataLoader(
        query_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=drop_last
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=drop_last
    )

    return train_loader, query_loader, gallery_loader, validation_loader
