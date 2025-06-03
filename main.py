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
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable channels_last memory format
        torch.backends.cudnn.enabled = True
        # Set optimal CUDA memory allocator
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
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
        batch_size=64,  # Increased for V100
        num_workers=6,  # One worker per core
        prefetch_factor=2  # Prefetch for optimization
    )
    print("Loaders ready!")
    print("Train batches (base):", len(base_train_loader))
    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    # === MODEL ===
    model = SiameseNetwork().to(device)

    # === TRAINING MODEL ===

    if training:
        # Try to compile the model with proper error handling
        try:
            # Check if we're on a CUDA device and if the CUDA version supports compilation
            if device.type == 'cuda':
                if torch.cuda.get_device_capability()[0] >= 7:  # Check for Volta or newer
                    print("Attempting to compile model with max-autotune...")
                    model = torch.compile(model, mode="max-autotune")
                    print("Model compilation successful!")
                else:
                    print("Warning: Your GPU does not support model compilation. Running without compilation.")
            else:
                print("Warning: Model compilation is only supported on CUDA devices. Running without compilation.")
        except Exception as e:
            print(f"Warning: Model compilation failed with error: {str(e)}")
            print("Continuing without model compilation...")

        train_siamese(
            model=model,
            train_loader=base_train_loader,
            val_loader=None,
            optimizer_type="adam",
            learning_rate=0.001,
            weight_decay=1e-4,
            epochs=1,
            gradient_accumulation_steps=2
        )

        # Save uncompiled model state
        torch.save(model.state_dict(), "model_repository/siamese_model.pth")
        print("Model saved successfully!")

    # === EVALUATION ===
    # Load model state and create new model instance
    model = SiameseNetwork().to(device)  # Create fresh model instance
    model.load_state_dict(torch.load("model_repository/siamese_model.pth", map_location=device))
    model.eval()  # Set to evaluation mode
    
    results = evaluate_siamese(model, query_loader, gallery_loader, device)

    print(f"🔍 Submitting results: {results}")
    res = submit(results, "Simple Guys")
    print(f"🔍 Submission result: {res}")

if __name__ == "__main__":
    run(training=True)

