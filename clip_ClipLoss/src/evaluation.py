import os
import torch
import glob
import re
import json
from tqdm import tqdm # Import tqdm for progress bars in embedding extraction
import clip

# Import your custom modules and config needed for evaluation
from config import config
from src.finetuned_clip import FineTunedCLIP # Needed if loading model from scratch or for type hints
from src.extract_embeddings_clip import extract_clip_embeddings
from src.results import get_top_k, load_images_from_folder
from torchvision import transforms

def extract_epoch_from_path(path):
    # This helper function is fine as it is.
    match = re.search(r"model_epoch_(\d+)\.pt", path) # Assuming model_epoch_X.pt from all_weights
    if not match: # Also check for weights_epoch_X.pth if needed for different loading scenarios
        match = re.search(r"weights_epoch_(\d+)\.pth", path)
    return int(match.group(1)) if match else None

# --- NEW: This is the function you should extract/create ---
def perform_retrieval_and_save_results(
    model: FineTunedCLIP, # Expects an instance of your FineTunedCLIP model
    query_dir: str,
    gallery_dir: str,
    top_k: int,
    distance_metric: str,
    device: torch.device,
    batch_size: int,
    epoch_number: int # Parameter to name the output JSON
):
    """
    Performs image retrieval for given query and gallery datasets using the provided model
    and saves the top-k results to a JSON file.

    Args:
        model: The trained FineTunedCLIP model.
        query_dir: Directory containing query images.
        gallery_dir: Directory containing gallery images.
        top_k: Number of top results to retrieve.
        distance_metric: Metric for similarity ("cosine" or "euclidean").
        device: Device to run computations on.
        batch_size: Batch size for embedding extraction.
        epoch_number: The current epoch number, used for naming the output JSON file.
    """
    model.eval() # Set model to evaluation mode

    # Prepare Test Data Preprocessing (should match training)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # Load query and gallery images
    print("   Loading query images for evaluation...")
    query_paths, query_images = load_images_from_folder(query_dir, transform=preprocess)
    print("   Loading gallery images for evaluation...")
    gallery_paths, gallery_images = load_images_from_folder(gallery_dir, transform=preprocess)
    print(f"   Loaded {len(query_images)} query images and {len(gallery_images)} gallery images.")

    # Extract Embeddings
    with torch.no_grad():
        print("   Extracting query features...")
        query_features = extract_clip_embeddings(model, query_images, device, batch_size=batch_size)
        print("   Extracting gallery features...")
        gallery_features = extract_clip_embeddings(model, gallery_images, device, batch_size=batch_size)
    print("   Embeddings extracted.")

    # Compute Retrieval Results
    print("   Getting top-k results...")
    result = get_top_k(
        query_embeds=query_features,
        gallery_embeds=gallery_features,
        gallery_paths=gallery_paths,
        query_paths=query_paths,
        k=top_k,
        distance=distance_metric
    )

    # Save Results
    results_dir = os.path.join("repository", "results")
    os.makedirs(results_dir, exist_ok=True) # Ensure this directory exists
    output_path = os.path.join(results_dir, f"retrieval_results_epoch_{epoch_number}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Retrieval results for epoch {epoch_number} saved to {output_path}")

# --- Standalone Execution Block ---
if __name__ == "__main__":
    print("Running src/evaluation.py in standalone mode.")
    
    cfg = config # Load configuration
    
    # 1. Load the base CLIP model and its preprocessing function
    clip_model_base, preprocess_base = clip.load("ViT-B/32", device=cfg.device)

    # 2. Specify which trained model to load for evaluation
    # You need to manually specify a path to one of your saved full models (model_epoch_X.pt)
    # For example, let's load the model from epoch 1
    TARGET_EPOCH_TO_EVALUATE = 1 # Change this to the epoch you want to test
    model_to_load_path = os.path.join("repository", "all_weights", f"model_epoch_{TARGET_EPOCH_TO_EVALUATE}.pt")

    if not os.path.exists(model_to_load_path):
        print(f"Error: Model checkpoint not found at {model_to_load_path}. "
              "Please train your model first or specify an existing path.")
        exit(1)

    try:
        # Load the entire FineTunedCLIP model object
        print(f"Loading model from {model_to_load_path}...")
        model_for_standalone_eval = torch.load(model_to_load_path, map_location=cfg.device, weights_only=False)
        model_for_standalone_eval.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from {model_to_load_path}: {e}")
        exit(1)

    # 3. Call the evaluation function
    perform_retrieval_and_save_results(
        model=model_for_standalone_eval,
        query_dir=cfg.query_dir,
        gallery_dir=cfg.gallery_dir,
        top_k=cfg.top_k,
        distance_metric=cfg.distance_metric,
        device=cfg.device,
        batch_size=cfg.batch_size,
        epoch_number=TARGET_EPOCH_TO_EVALUATE, # Use the specified epoch number
        preprocess_func=preprocess_base # Pass the preprocess function
    )
    print(f"\nStandalone evaluation for Epoch {TARGET_EPOCH_TO_EVALUATE} complete. Check results in 'repository/results/'.")



