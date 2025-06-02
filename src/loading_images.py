from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def load_data(train_dir, batch_size):
    transform = get_transforms()
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.classes
