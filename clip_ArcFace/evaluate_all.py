import os
import torch
import glob
import re
from src.data_loading import datasets
from src.model import ViTClassifier
from src.test import fast_test, save_results_to_json

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def extract_epoch_from_path(path):
    match = re.search(r"model_epoch_(\d+)\.pt", path)
    return int(match.group(1)) if match else None

def main():
    # == Config ==
    train_directory = "data/train"
    query_directory = "data/test/query"
    gallery_directory = "data/test/gallery"
    validation_directory = "data/validation"
    batch_size = 126
    num_worker = 4
    weights_dir = "repository/all_weights"
    results_dir = "repository/results"
    k_retrieval = 10

    device = get_device()
    print(f"✅ Using device: {device}")

    # == Load dataset once ==
    train_loader, query_loader, gallery_loader, _ = datasets(
        training_dir=train_directory,
        query_dir=query_directory,
        gallery_dir=gallery_directory,
        validation_dir=validation_directory,
        batch_size=batch_size,
        num_workers=num_worker,
        drop_last=True
    )
    print("✅ Dati caricati con successo")
    num_classes = len(train_loader.dataset.classes)

    os.makedirs(results_dir, exist_ok=True)

    checkpoint_paths = sorted(glob.glob(os.path.join(weights_dir, "model_epoch_*.pt")))
    if not checkpoint_paths:
        print("❌ Nessun file di pesi trovato.")
        return

    for checkpoint_path in checkpoint_paths:
        epoch_num = extract_epoch_from_path(checkpoint_path)
        if epoch_num is None:
            continue

        print(f"\n🔄 Analisi del checkpoint: epoch {epoch_num}")

        model = ViTClassifier(num_classes=num_classes, freeze=False, device=device).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        result = fast_test(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            k=k_retrieval,
            device=device
        )

        output_path = os.path.join(results_dir, f"model_epoch_{epoch_num}.json")
        save_results_to_json(result, output_path)
        print(f"✅ Risultato salvato in {output_path}")

if __name__ == "__main__":
    main()
