from src.data_loading import retrival_data_loading
from src.model import resnet50
from src.training_loop import training_loop
from src.embedding import extract_embeddings
from src.results import compute_results
from src.evaluation import evaluation
from src.triplet_dataset import TripletDataset
from config import ModelConfig, TrainingConfig
import torch

def run(training=True):
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # === DEVICE ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === DATA LOADING ===
    base_train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root="data_example_animal/train",
        query_data_root="data_example_animal/test/query",
        gallery_data_root="data_example_animal/test/gallery",
        batch_size=train_cfg.batch_size
    )
    print("Loaders pronti!")
    print("Train batches (base):", len(base_train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    if training:
        # === MINING MODEL (eval only) ===
        mining_model = resnet50(
            model_cfg=model_cfg,
            pretrained=True
        ).to(device)
        mining_model.eval()

        # === TRAINING MODEL ===
        model = resnet50(
            model_cfg=model_cfg,
            pretrained=True
        ).to(device)

        if train_cfg.compiled:
            model = torch.compile(model, backend="aot_eager")

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
            num_workers=4
        )

        # === TRAINING ===
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

        # === SAVE MODEL ===
        torch.save(model._orig_mod.state_dict(), train_cfg.model_save_path)
        print("Modello salvato")

    # === EVALUATION MODEL ===
    model = resnet50(
        model_cfg=model_cfg,
        pretrained=True
    )
    model.load_state_dict(torch.load(train_cfg.model_save_path, map_location=device, weights_only=True))
    model = model.to(device)

    # === EMBEDDING EXTRACTION ===
    query_df = extract_embeddings(query_loader, model, device)
    gallery_df = extract_embeddings(gallery_loader, model, device)

    # === COMPUTE RESULTS ===
    results = compute_results(query_df, gallery_df, metric="euclidean")

    # === EVALUATION ===
    evaluation(results, "data_example_rota/query_to_gallery_mapping.json")

if __name__ == "__main__":
    run(training=True)
