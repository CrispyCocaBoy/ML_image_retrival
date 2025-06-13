import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset # Import Subset for dataset splitting
import random # Import random for shuffling indices
from types import SimpleNamespace # Make sure SimpleNamespace is imported if config is directly defined here or for clarity

# Import your configuration from a separate config.py file
from config import config

# Import your custom modules
from src.finetuned_clip import FineTunedCLIP
from src.image_text_dataset import ImageTextDataset
from src.training_loop import train_clip_hybrid
from src.extract_embeddings_clip import extract_clip_embeddings
from src.results import get_top_k, load_images_from_folder

# This function checks if a pre-trained model exists or if training is forced.
# It then either loads an existing model or trains a new one and saves it.
# Finally, it returns the ready-to-use model instance.
def check_and_train_model():
    model_path = "finetuned_clip.pth" # Define the path for saving/loading the model
    device = config.device
    
    # Initialize the model. This instance will be either trained or loaded into.
    # freeze_clip is set to True, meaning only the projection layers will be fine-tuned.
    model = FineTunedCLIP(device=device, embed_dim=config.embedding_dim, freeze_clip=True)

    # Logic to decide whether to train from scratch or load existing model
    if config.force_train or not os.path.exists(model_path):
        print("🚀 Starting fine-tuning of CLIP model with hybrid loss...")
        
        # Define preprocessing for the training dataset
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        
        # Prepare the full dataset
        full_train_dataset = ImageTextDataset(config.train_dir, transform=preprocess)
        
        # --- Split dataset into training and validation sets ---
        dataset_size = len(full_train_dataset)
        validation_split = 0.15 # 15% for validation, adjust as needed
        indices = list(range(dataset_size))
        random.shuffle(indices) # Shuffle indices for random split
        
        split = int(validation_split * dataset_size)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        
        # Create dataloaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) # No need to shuffle validation
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")
        
        # Perform the actual training, passing both train and val dataloaders
        train_clip_hybrid(model, train_loader, val_loader, device, epochs=config.epochs, lr=config.learning_rate)
        
        # Save the state dictionary of the newly trained model to the specified path
        torch.save(model.state_dict(), model_path)
        print(f"✅ Trained model saved to {model_path}")
    else:
        # If the model file exists and force_train is False, load the existing model
        print("✅ Fine-tuned model already exists. Loading it.")
        # Load the state dictionary from the file into the pre-initialized model object
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Always return the fully prepared model object, whether trained or loaded
    return model

# Main execution function
def main():
    device = config.device

    # Get the model (either trained or loaded) by calling the helper function
    model = check_and_train_model()
    
    # Set the model to evaluation mode. This is crucial for inference,
    # as it disables dropout, batch normalization updates, etc., which are
    # only needed during training.
    model.eval()

    # === Prepare Test Datasets (for query and gallery images) ===
    # Preprocessing for query and gallery images should match training preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # Load images and their paths from the query and gallery directories
    print("Loading query images...")
    query_paths, query_images = load_images_from_folder(config.query_dir, transform=preprocess)
    print("Loading gallery images...")
    gallery_paths, gallery_images = load_images_from_folder(config.gallery_dir, transform=preprocess)

    # Extract embeddings for the loaded images using the fine-tuned CLIP model
    print("Extracting query features...")
    with torch.no_grad(): # Disable gradient calculations for inference, saves memory and speeds up
        query_features = extract_clip_embeddings(model, query_images, device)
    print("Extracting gallery features...")
    with torch.no_grad():
        gallery_features = extract_clip_embeddings(model, gallery_images, device)

    # Perform image retrieval: get the top-k most similar gallery images for each query image
    print("Getting top-k results...")
    results = get_top_k(query_features, gallery_features, gallery_paths, query_paths,
                        k=config.top_k, distance=config.distance_metric)

    # Save the retrieval results to a JSON file
    output_filename = "retrieval_results.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Ranking saved in {output_filename}")

# Entry point of the script
if __name__ == "__main__":
    main()

