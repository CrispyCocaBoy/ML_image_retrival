from src.data_loading import retrival_data_loading
from src.model import resnet50, resnet50v2, resnet50v3, clipper
from src.training_loop import training_loop
from src.embedding import extract_embeddings
from src.results import compute_results
from src.evaluation import evaluation
from src.triplet_dataset import TripletDataset
from src.training_loop_early_stop import *
from config import DataConfig, ModelConfig, TrainingConfig
from multiprocessing import cpu_count
import torch

MML_model = resnet50v3  # Change index to select model, e.g., resnet50v2, resnet50, clipper

def run(training=True):
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

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

    if training:
        # === MINING MODEL (solo per costruire triplette) ===
        mining_model = MML_model(
            model_cfg=model_cfg,
            pretrained=True).to(device)
        mining_model.eval()

        # === TRAINING MODEL ===
        model = MML_model(
            model_cfg=model_cfg,
            pretrained=True ).to(device)

        # Compilazione opzionale
        if train_cfg.compiled:
            try:
                model = torch.compile(model, backend="aot_eager")
            except Exception as e:
                print("torch.compile fallita, continuo senza compilazione:", e)

        # === TRIPLET DATASET ===
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

        # === TRAINING ===
        training_loop(
            train_loader=train_loader,
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            ground_truth_path=DataConfig.ground_truth_path,
            early_stopping_patience=5,
            model_save_path=TrainingConfig.model_save_path,
            epochs=train_cfg.epochs,
        )

        # === SALVATAGGIO ROBUSTO ===
        save_path = train_cfg.model_save_path
        if hasattr(model, "_orig_mod"):
            torch.save(model._orig_mod.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print(f"Modello salvato in: {save_path}")

    # === EVALUATION MODEL ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps"):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = MML_model(
        model_cfg=model_cfg,
        pretrained=False)
    model.load_state_dict(torch.load(train_cfg.model_save_path, map_location=device))
    model = model.to(device)

    # === EMBEDDING EXTRACTION ===
    query_df = extract_embeddings(query_loader, model, device)
    gallery_df = extract_embeddings(gallery_loader, model, device)

    # === COMPUTE RESULTS ===
    results = compute_results(query_df, gallery_df, metric="euclidean")

    # === EVALUATION ===
    evaluation(results, "data_example_animal/query_to_gallery_mapping.json")
'''
    # === VISUALIZZAZIONE ===
    show_image_results(
        results,
        top_k=3,
        max_queries=5,
        base_query_dir=data_cfg.query_data_root,
        base_gallery_dir=data_cfg.gallery_data_root
    )
'''
if __name__ == "__main__":
    run(training=True)  # Set to False to skip training and only evaluate

