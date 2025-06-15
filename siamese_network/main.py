# == Packages ==
import torch
from src.data_loading import datasets
from src.model import *
from src.train import train_loop
from src.test import *

# == Definizione variabili globali ==
train_directory = "data/train"
query_directory = "data/test/query"
gallery_directory = "data/test/gallery"
validation_directory = "data/validation"
batch_size = 16
seed = 1
epochs = 35
num_pairs = 5000
num_worker = 4

# == Definizione variabili allenamento ==
learning_rate = 1e-4  # It will get modified by scheduler
optimizer = "adam"
margin = 0.5



# == Run del modello ==
def run(training):
    # == device option ==
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps"):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # == Creation of the datasets ==
    train, validation, query, gallery = datasets(
        training_dir=train_directory,
        query_dir=query_directory,
        gallery_dir=gallery_directory,
        batch_size=batch_size,
        seed=seed,
        num_pairs=num_pairs,
        num_workers=num_worker,
        num_val = batch_size * 2
    )
    print(f"Dati caricati con successo")

    # == Initilize the model ==
    model = SiameseNetwork(freeze=False, device = device).to(device)

    if training == True:
        train_loop(
            model=model,
            dataloader_train = train,
            dataloader_validation = validation,
            device=device,
            epochs=epochs,
            margin=margin,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
            optimizer_name=optimizer,  # "SGD" or "adam"
            use_checkpoint=True,
            early_stops=True,
            patience=5,
            save_weights=True
        )

    # == Evaluation ==
    ## Riprende i pesi savlati
    #model.load_state_dict(torch.load("repository/best_model/model.pt", map_location=device))
    model.eval()  # Modalita valutazione
    result = fast_test(model=model,
                       query_loader=query,
                       gallery_loader=gallery,
                       device=device)

    submit(result)
    show_retrieved_images(result, query_dir=query_directory, gallery_dir=gallery_directory)
    save_results_to_json(result, "retrieval_results.json")


if __name__ == "__main__":
    run(training=False)