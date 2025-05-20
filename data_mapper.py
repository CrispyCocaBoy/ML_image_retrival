import os
import shutil
import random
import json
from pathlib import Path
from collections import defaultdict


def prepare_dataset(base_dir, query_per_class=1, gt_per_query=5, gallery_pct=0.8, seed=42):
    random.seed(seed)

    base_dir = Path(base_dir)
    train_dir = base_dir / "train"
    test_query_dir = base_dir / "test/query"
    test_gallery_dir = base_dir / "test/gallery"
    query_mapping = {}

    # Creazione cartelle
    test_query_dir.mkdir(parents=True, exist_ok=True)
    test_gallery_dir.mkdir(parents=True, exist_ok=True)

    all_gallery_images = []
    all_classes = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    used_images = set()

    for cls_path in all_classes:
        cls_name = cls_path.name
        images = sorted(list(cls_path.glob("*.jpg")))
        if len(images) < query_per_class + gt_per_query + 1:
            print(f"❗ Classe '{cls_name}' ha troppe poche immagini, saltata.")
            continue

        # Random shuffle per selezione
        random.shuffle(images)
        query_img = images.pop()  # 1 query
        gt_imgs = [images.pop() for _ in range(gt_per_query)]  # 5 ground truth

        # Copia query
        shutil.copy(query_img, test_query_dir / query_img.name)
        used_images.add(query_img)
        used_images.update(gt_imgs)

        query_mapping[query_img.name] = [img.name for img in gt_imgs]

        # Resto disponibile per bilanciamento gallery
        all_gallery_images.extend([(img, cls_name) for img in images])

    # Bilanciamento: prendiamo una % (gallery_pct) delle immagini totali
    total_per_class = defaultdict(list)
    for img, cls_name in all_gallery_images:
        total_per_class[cls_name].append(img)

    # Numero target immagini gallery totali
    total_images = sum(len(v) for v in total_per_class.values())
    gallery_target = int(total_images * gallery_pct)

    # Distribuzione bilanciata tra classi
    num_classes = len(total_per_class)
    per_class_target = max(1, gallery_target // num_classes)

    gallery_final = []

    for cls_name, imgs in total_per_class.items():
        selected = imgs[:per_class_target]
        for img in selected:
            shutil.copy(img, test_gallery_dir / img.name)
            used_images.add(img)
            gallery_final.append(img)

    # Pulizia immagini usate dal train
    for img in used_images:
        try:
            os.remove(img)
        except FileNotFoundError:
            continue

    # Salva mapping
    with open(base_dir / "query_to_gallery_mapping.json", "w") as f:
        json.dump(query_mapping, f, indent=2)

    print(f"✅ Dataset pronto.")
    print(f"📁 Query: {len(query_mapping)}")
    print(f"📁 Gallery: {len(gallery_final)}")
    print(f"📁 Ground truth per query: {gt_per_query} immagini ciascuna")


# Usa lo script
if __name__ == "__main__":
    prepare_dataset("data_example")