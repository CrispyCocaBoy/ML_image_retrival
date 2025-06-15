import os
import random
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pool di trasformazioni atomiche
trasformazioni_base = [
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomVerticalFlip(p=1.0),
]

# Costruisce una sequenza casuale di 1-3 trasformazioni
def get_combinazione_casuale():
    num = random.randint(1, 3)
    scelte = random.sample(trasformazioni_base, k=num)
    return transforms.Compose([
        transforms.Resize((256, 256)),
        *scelte,
        transforms.Resize((224, 224)),  # output coerente
    ])

def augmenta_una(classe_path, img, suffix):
    transform = get_combinazione_casuale()
    nuova_img = transform(img)
    nuovo_nome = f"aug_{suffix}_{random.randint(1000,9999)}.jpg"
    nuova_img.save(os.path.join(classe_path, nuovo_nome))

def augmenta_per_class(classe_path, immagini, num_target=11):
    immagini_valide = [
        Image.open(os.path.join(classe_path, img)).convert("RGB")
        for img in immagini
    ]
    num_attuali = len(immagini)
    da_creare = num_target - num_attuali
    if da_creare <= 0:
        return

    print(f"→ Classe '{os.path.basename(classe_path)}': {num_attuali} immagini, genero {da_creare} augmentazioni")
    for i in range(da_creare):
        img = random.choice(immagini_valide)
        augmenta_una(classe_path, img, suffix=i)

def processa_dataset(train_path: str, num_target: int = 11):
    classi = [
        c for c in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, c))
    ]
    for classe in tqdm(classi, desc="Processo classi"):
        classe_path = os.path.join(train_path, classe)
        immagini = [
            f for f in os.listdir(classe_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        augmenta_per_class(classe_path, immagini, num_target=num_target)

if __name__ == "__main__":
    # Risale automaticamente alla radice del progetto
    base_dir = Path(__file__).resolve().parent.parent
    # Costruisce il path alla cartella train dentro data_original
    train_dir = base_dir / "data" / "train"

    # Esegui il processo di augmentazione
    processa_dataset(str(train_dir), num_target=10)
