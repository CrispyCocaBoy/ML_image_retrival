import os
import torch
import glob
import re
import json

# Import your current project modules and config
from config import config
# from src.finetuned_clip import FineTunedCLIP # Not strictly needed for loading full model, but can be kept for clarity
from src.extract_embeddings_clip import extract_clip_embeddings
from src.results import get_top_k, load_images_from_folder
from torchvision import transforms

def extract_epoch_from_path(path):
    """
    Extracts the epoch number from a full model checkpoint path (e.g., 'model_epoch_10.pt').
    """
    # Regex updated to match .pt files saved as 'model_epoch_X.pt'
    match = re.search(r"model_epoch_(\d+)\.pt", path)
    return int(match.group(1)) if match else None

def main():
    # == Configuration from config.py ==
    query_directory = config.query_dir
    gallery_directory = config.gallery_dir
    batch_size = config.batch_size
    k_retrieval = config.top_k
    distance_metric = config.distance_metric

    # Target the all_weights directory for .pt files (full models)
    weights_dir = "repository/all_weights"
    results_dir = "repository/results"

    device = config.device
    print(f"✅ Using device: {device}")

    # == Prepare Test Data Preprocessing ==
    # This preprocessing pipeline should exactly match what was used during training
    # for consistency when extracting embeddings.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # == Load Query and Gallery Images ==
    # Load all query and gallery images once, as they remain constant for all checkpoints.
    print("Loading query images for evaluation...")
    query_paths, query_images = load_images_from_folder(query_directory, transform=preprocess)
    print("Loading gallery images for evaluation...")
    gallery_paths, gallery_images = load_images_from_folder(gallery_directory, transform=preprocess)
    print(f"✅ Loaded {len(query_images)} query images and {len(gallery_images)} gallery images.")

    # == Setup Results Directory ==
    # Create the directory where the JSON results will be saved if it doesn't exist.
    os.makedirs(results_dir, exist_ok=True)

    # == Collect Checkpoint Paths ==
    # Find all saved full model files (.pt) in the all_weights directory.
    # The list is sorted to process epochs in order.
    checkpoint_paths = sorted(glob.glob(os.path.join(weights_dir, "model_epoch_*.pt")))
    
    if not checkpoint_paths:
        print(f"❌ No full model files found in '{weights_dir}'. Please ensure training has occurred and .pt files are saved (e.g., 'model_epoch_1.pt').")
        return

    # == Iterate and Evaluate Each Checkpoint ==
    for checkpoint_path in checkpoint_paths:
        epoch_num = extract_epoch_from_path(checkpoint_path)
        if epoch_num is None:
            print(f"Skipping {checkpoint_path}: could not extract epoch number. Check filename format.")
            continue

        print(f"\n🔄 Analyzing checkpoint: epoch {epoch_num} from {checkpoint_path}")

        # Load the entire model directly.
        # weights_only=False is crucial for PyTorch 2.6+ to load full model objects (not just state_dict).
        # It's safe here because you generated these files yourself.
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Set the model to evaluation mode. This disables dropout, batch norm updates, etc.,
        # ensuring consistent results during inference.
        model.eval()

        # == Extract Embeddings ==
        # Extract features from query and gallery images using the current model checkpoint.
        # `torch.no_grad()` is used to disable gradient computation, saving memory and speeding up inference.
        with torch.no_grad():
            print("   Extracting query features...")
            query_features = extract_clip_embeddings(model, query_images, device, batch_size=batch_size)
            print("   Extracting gallery features...")
            gallery_features = extract_clip_embeddings(model, gallery_images, device, batch_size=batch_size)
        print("   Embeddings extracted.")

        # == Compute Retrieval Results ==
        # Use the extracted embeddings to find the top-k most similar gallery images for each query.
        result = get_top_k(
            query_embeds=query_features,
            gallery_embeds=gallery_features,
            gallery_paths=gallery_paths,
            query_paths=query_paths,
            k=k_retrieval,
            distance=distance_metric
        )

        # == Save Results ==
        # Define the output JSON file path and save the retrieval results.
        output_path = os.path.join(results_dir, f"retrieval_results_epoch_{epoch_num}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Retrieval results for epoch {epoch_num} saved to {output_path}")

if __name__ == "__main__":
    main()
