import torch

from src.data_loading import retrival_data_loading
from src.siamsese import SiameseNetwork
from src.training_loop import train_siamese
from src.evaluation import evaluate_siamese
from src.submit_api import submit

train_data_root="data/train"
query_data_root="data/test/query"
gallery_data_root= "data/test/gallery"



def run(training=True):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Ottimizzazioni CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif hasattr(torch.backends, "mps"):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # === DATA LOADING ===
    base_train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root=train_data_root,
        query_data_root=query_data_root,
        gallery_data_root=gallery_data_root,
        batch_size=64,  # Aumentato per V100
        num_workers=6,  # Un worker per core
        prefetch_factor=2  # Prefetch per ottimizzare il caricamento
    )
    print("Loaders pronti!")
    print("Train batches (base):", len(base_train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    # === MODEL ===
    model = SiameseNetwork(backbone="resnet18").to(device)  # Usiamo ResNet18 per velocità

    # === TRAINING MODEL ===
    if training == True:

        # Compilation del modello
        try:
            model = torch.compile(model, mode="max-autotune")  # Ottimizzazione massima
        except Exception as e:
            print("torch.compile fallita, continuo senza compilazione:", e)

        # === TRIPLET DATASET ===
        train_loader, query_loader, gallery_loader = retrival_data_loading(
            train_data_root=train_data_root,
            query_data_root=query_data_root,
            gallery_data_root=gallery_data_root,
            batch_size=64,  # Aumentato per V100
            num_workers=6,  # Un worker per core
            prefetch_factor=2  # Prefetch per ottimizzare il caricamento
        )

        # === TRAINING ===
        train_siamese(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer_type="adam",
            learning_rate=0.001,
            weight_decay=1e-4,
            epochs=1,
        )

        # === SALVATAGGIO ROBUSTO ===
        torch.save(model.state_dict(), "model_repository/siamese_model.pth")
        print(f"✅ Modello salvato in: model_repository/siamese_model.pth")
    
    # === EVALUATION MODEL ===
    # Recall of the model from the repository
    model.load_state_dict(torch.load("model_repository/siamese_model.pth"))
    results = evaluate_siamese(model, query_loader, gallery_loader, device)
    
    # Result to server 
    print(f"🔍 Eseguo la submit con i risultati: {results}")
    res = submit(results, "Simple Guys")
    print(f"🔍 Risultato della submit: {res}")

if __name__ == "__main__":
    run(training=True)

