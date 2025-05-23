from src.data_loading import *
from src.model import resnet50v2
from src.training_loop import training_loop
from src.embedding import extract_embeddings
from src.results import compute_results
from src.evaluation import evaluation

import os
import torch


def run(training = True):

    # Data_loading
    train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root="data_example_animal/train",
        query_data_root="data_example_animal/test/query",
        gallery_data_root="data_example_animal/test/gallery",
        triplet_loss=True,
        batch_size=32
    )
    print("Loaders pronti!")
    print("Train batches:", len(train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    if training == True:
        # Modello utilizzato
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"📡 In uso: {device}")
        model = resnet50v2(pretrained=True, embedding_dim=128).to(device)
        model = torch.compile(model, backend="aot_eager")

        # training del modello
        training_loop(
            train_loader=train_loader,
            model=model,
            optimizer_type = "adam",
            epochs=10,
            loss="triplet"
        )

        # Salvare il modello allenato
        torch.save(model._orig_mod.state_dict(), "model_repository/resnet_triplet.pth")
        print("Modello salvato")

    # Test
    ## Alcune volte vogliamo solo eseguire il test senza allenamento quindi prendiamo quello che abbiamo già allenato
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"📡 In uso: {device}")
    model = resnet50v2(pretrained=False, embedding_dim=128)
    model.load_state_dict(torch.load("model_repository/resnet_triplet.pth", map_location=device))
    model = model.to(device)

    ## Estraiamo gli embeddings dai due dataset
    query_df = extract_embeddings(query_loader, model, device)
    gallery_df = extract_embeddings(gallery_loader, model, device)

    ## Calcoliamo le immagini più vicine
    results = compute_results(query_df, gallery_df, metric="euclidean")

    ## Confronto con la ground truth 
    evaluation(results, "data_example_animal/query_to_gallery_mapping.json")


if __name__ == "__main__":
    run(training=True)


