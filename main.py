import torch

from src.data_loading import retrival_data_loading
from src.siamsese import SiameseNetwork
from src.training_loop import train_siamese
from src.evaluation import evaluate_siamese
from src.submit_api import submit




def run(training=True):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps"):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # === DATA LOADING ===
    base_train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root="data_competition/train",
        query_data_root="data_competition/test/query",
        gallery_data_root= "data_competition/test/gallery",
        batch_size= 2
    )
    print("Loaders pronti!")
    print("Train batches (base):", len(base_train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    # === MODEL ===
    model = SiameseNetwork().to(device)

    # === TRAINING MODEL ===
    if training == True:

        # Compilation del modello
        try:
            model = torch.compile(model, backend="eager")
        except Exception as e:
            print("torch.compile fallita, continuo senza compilazione:", e)

        # === TRIPLET DATASET ===
        train_loader, query_loader, gallery_loader = retrival_data_loading(
            train_data_root="data_example_rota/train",
            query_data_root="data_example_rota/test/query",
            gallery_data_root= "data_example_rota/test/gallery",
            batch_size= 2)

        # === TRAINING ===
        train_siamese(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer_type="adam",
            learning_rate=0.1,
            weight_decay=1e-4,
            epochs=2,
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

