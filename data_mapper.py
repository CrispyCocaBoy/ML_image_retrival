import os
import shutil
import random
import json
from pathlib import Path
from collections import defaultdict

def prepare_dataset(base_dir, query_per_class=1, gt_per_query=3, seed=42):
    random.seed(seed)

    base_dir = Path(base_dir)
    train_dir = base_dir / "train"
    test_query_dir = base_dir / "test/query"
    test_gallery_dir = base_dir / "test/gallery"
    query_mapping = {}

    # Creazione cartelle
    test_query_dir.mkdir(parents=True, exist_ok=True)
    test_gallery_dir.mkdir(parents=True, exist_ok=True)

    all_classes = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    used_images = set()

    for cls_path in all_classes:
        cls_name = cls_path.name
        images = sorted(list(cls_path.glob("*.jpg")))

        if len(images) < query_per_class + gt_per_query:
            print(f"❗ Classe '{cls_name}' ha troppe poche immagini, saltata.")
            continue

        # Random shuffle per selezione
        random.shuffle(images)
        query_img = images.pop()  # 1 query
        gt_imgs = [images.pop() for _ in range(gt_per_query)]  # 3 ground truth

        # Copia query
        shutil.copy(query_img, test_query_dir / query_img.name)
        used_images.add(query_img)

        # Copia ground truth nella gallery
        for img in gt_imgs:
            shutil.copy(img, test_gallery_dir / img.name)
            used_images.add(img)

        query_mapping[query_img.name] = [img.name for img in gt_imgs]

    # Rimuove le immagini usate dal train
    for img in used_images:
        try:
            if img.exists():
                img.unlink()
        except Exception as e:
            print(f"Errore durante l'eliminazione di {img}: {e}")

    # Salva mapping
    with open(base_dir / "query_to_gallery_mapping.json", "w") as f:
        json.dump(query_mapping, f, indent=2)

    print(f"✅ Dataset pronto.")
    print(f"📁 Query: {len(query_mapping)}")
    print(f"📁 Gallery: {len(list(test_gallery_dir.glob('*.jpg')))}")
    print(f"📁 Ground truth per query: {gt_per_query} immagini ciascuna")

    # Report per classe (solo GT)
    gallery_class_count = defaultdict(int)
    for cls_path in all_classes:
        cls_name = cls_path.name
        if len(list(cls_path.glob("*.jpg"))) < query_per_class + gt_per_query:
            continue
        gallery_class_count[cls_name] = gt_per_query

    print("📊 Immagini in gallery per categoria (solo ground truth):")
    for cls, count in gallery_class_count.items():
        print(f"  🐾 {cls}: {count}")

# Usa lo script
if __name__ == "__main__":
    prepare_dataset("data_example_animal")

