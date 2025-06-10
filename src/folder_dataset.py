from torch.utils.data import Dataset
from PIL import Image
import os

class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
class InferenceFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(class_path, fname)
                    self.samples.append(full_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path  # Return path instead of label
    
class FlatInferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        print(f"Looking for images in: {root_dir}")
        
        if not os.path.exists(root_dir):
            print(f"ERROR: Directory {root_dir} does not exist!")
            return
            
        for fname in os.listdir(root_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                full_path = os.path.join(root_dir, fname)
                self.samples.append(full_path)
                
        print(f"Found {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path