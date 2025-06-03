# Evaluate the model 

import torch
from tqdm import tqdm
from pathlib import Path
import json

def evaluate_siamese(model, query_loader, gallery_loader, device, top_k=5):
    """
    Evaluate the Siamese network by comparing query images with gallery images.
    
    Args:
        model: Siamese network model
        query_loader: DataLoader for query images
        gallery_loader: DataLoader for gallery images
        device: Device to run inference on
        top_k: Number of top matches to return for each query
    
    Returns:
        Dictionary mapping query image names to lists of top-k matching gallery image names
    """
    model.eval()
    results = {}
    
    # Get gallery images
    gallery_images = []
    gallery_names = []
    
    print("Loading gallery images...")
    for images, names in tqdm(gallery_loader):
        gallery_images.append(images)
        gallery_names.extend(names)
    
    gallery_images = torch.cat(gallery_images, dim=0)
    
    # Compare each query with gallery
    print("Comparing queries with gallery...")
    with torch.no_grad():
        for query_img, query_name in tqdm(query_loader):
            query_img = query_img.to(device)
            gallery_batch = gallery_images.to(device)
            
            # Expand query image to match gallery batch size
            query_expanded = query_img.expand(gallery_batch.size(0), -1, -1, -1)
            
            # Get distances directly from model
            distances = model(query_expanded, gallery_batch)
            
            # Get top-k closest matches
            top_k_distances, top_k_indices = torch.topk(distances, k=top_k, largest=False)
            
            # Store results
            results[query_name[0]] = [gallery_names[idx] for idx in top_k_indices]
    
    return results

def save_results(results, output_path):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary of results from evaluate_siamese
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

