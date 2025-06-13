# == Packages ==
from src.data_loading import datasets
from src.model import *
from src.train import *
from src.test import *


# == Definizione variabili globali ==
train_directory = "data/train"
query_directory = "data/test/query"
gallery_directory = "data/test/gallery"
validation_directory = "data/validation"
batch_size = 126
epochs = 20
num_worker = 4

# == Definizione variabili allenamento ==
learning_rate = 0.00001
optimizer = "adamw"


# == Run del modello ==
def run(training):
    # == device option ==
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # == Creation of the datasets ==
    train, query, gallery, validation = datasets(
        training_dir=train_directory,
        query_dir=query_directory,
        gallery_dir=gallery_directory,
        validation_dir=validation_directory,
        batch_size=batch_size,
        num_workers=num_worker,
        drop_last=True
    )
    print(f"Dati caricati con successo")

    # Embedding
    num_classes = len(train.dataset.classes)
    print(f"📦 Numero di classi nel dataset di training: {num_classes}")

    # == Initilize the model ==
    model = ViTClassifier(num_classes=num_classes, freeze=False, device=device).to(device)

    if training == True:
        train_loop(
            model=model,
            train_loader=train,
            validation_loader=validation,
            device=device,
            epochs=epochs,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-7,
            optimizer_name=optimizer,  # "SGD" or "adam"
            use_checkpoint=True,
            early_stops=True,
            patience=5,
            save_weights=True
        )

        # == Evaluation ==
    ## Riprende i pesi savlati
    epoch_to_analyze = 28
    weights_path = f"repository/all_weights/model_epoch_{epoch_to_analyze}.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Modalita valutazione
    result = fast_test(model=model,
                       query_loader=query,
                       gallery_loader=gallery,
                       k=10,
                       device=device)

    # show_retrieved_images(result, query_dir= query_directory, gallery_dir = gallery_directory)
    output_filename = f"result_epoch_{epoch_to_analyze}.json"
    save_results_to_json(result, output_path=output_filename)


if __name__ == "__main__":
    run(training=True)