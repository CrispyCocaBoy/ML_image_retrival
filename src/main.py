import sys
from pathlib import Path

# Aggiunge 'src/' alla sys.path (dinamicamente)
sys.path.append(str(Path(__file__).resolve().parent))

from project.data_loading import retrival_data_loading

def run():
    train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root="data/train",
        query_data_root="data/query",
        gallery_data_root="data/gallery",
        triplet_loss=True,
        batch_size=32
    )
    print("Loaders pronti!")
    print("Train batches:", len(train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

if __name__ == "__main__":
    run()
