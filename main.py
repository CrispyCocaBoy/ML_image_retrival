import torch
from multiprocessing import cpu_count

from src.data_loading import retrival_data_loading
from src.model import build_model
from src.training_loop import training_loop
from src.embedding import extract_embeddings
from src.results import compute_results
from src.evaluation import evaluation
from src.triplet_dataset import TripletDataset
from src.show_images import show_image_results
from config import DataConfig, ModelConfig, TrainingConfig

def run(training=True):
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    # === DEVICE ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps"):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # === DATA LOADING ===
    base_train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root=data_cfg.train_data_root,
        query_data_root=data_cfg.query_data_root,
        gallery_data_root=data_cfg.gallery_data_root,
        batch_size=train_cfg.batch_size
    )
    print("Loaders pronti!")
    print("Train batches (base):", len(base_train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    # === TRAINING ===
    if training:
        # Mining model usato per costruire le triplette
        mining_model = build_model(model_cfg, pretrained=True).to(device)
        mining_model.eval()

        # Modello effettivo da allenare
        model = build_model(model_cfg, pretrained=True).to(device)

        # Opzionale: torch.compile()
        if train_cfg.compiled:
            try:
                model = torch.compile(model, backend="aot_eager")
            except Exception as e:
                print("⚠️ torch.compile fallita:", e)

        # TripletDataset
        train_dataset = TripletDataset(
            image_folder_dataset=base_train_loader.dataset,
            mining_strategy=train_cfg.mining_strategy,
            model=mining_model,
            device=device,
            margin=train_cfg.margin
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=min(8, cpu_count())
        )

        # Loop di training
        training_loop(
            train_loader=train_loader,
            model=model,
            optimizer_type=train_cfg.optimizer,
            epochs=train_cfg.epochs,
            loss=train_cfg.loss,
            lr=train_cfg.learning_rate,
            margin=train_cfg.margin,
            weight_decay=train_cfg.weight_decay
        )

        # Salvataggio del modello
        save_path = train_cfg.model_save_path
        torch.save(getattr(model, "_orig_mod", model).state_dict(), save_path)
        print(f"✅ Modello salvato in: {save_path}")

    # === EVALUATION ===
    model = build_model(model_cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(train_cfg.model_save_path, map_location=device))

    query_df = extract_embeddings(query_loader, model, device)
    gallery_df = extract_embeddings(gallery_loader, model, device)

    # Compute e Evaluate
    results = compute_results(query_df, gallery_df, metric=train_cfg.distance_metric)
    evaluation(results, "data_example_animal/query_to_gallery_mapping.json")

    # Visualizzazione dei risultati
    show_image_results(
        results,
        top_k=3,
        max_queries=5,
        base_query_dir=data_cfg.query_data_root,
        base_gallery_dir=data_cfg.gallery_data_root
    )

if __name__ == "__main__":
    run(training=True)
