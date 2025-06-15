import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple

def load_images_from_folder(folder: str, transform) -> Tuple[List[str], torch.Tensor]:
    """
    Loads images from a specified folder and applies a given transformation.

    This function scans the provided directory for common image file types,
    loads them using Pillow, converts them to RGB, applies the torchvision
    transformations, and stacks them into a single PyTorch tensor.

    Args:
        folder (str): The path to the directory containing the images.
        transform: A torchvision transform object (e.g., from `torchvision.transforms.Compose`)
                   to apply to each image.

    Returns:
        Tuple[List[str], torch.Tensor]: A tuple containing:
            - paths (List[str]): A list of full file paths for the loaded images.
            - images (torch.Tensor): A batched tensor of the transformed images
                                     (shape: [N, C, H, W]).
    """
    # List all files in the folder and filter for common image extensions.
    paths = [os.path.join(folder, fname) for fname in os.listdir(folder)
             if fname.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Load each image, convert to RGB, apply the transform, and store with its path.
    # .convert("RGB") ensures consistent 3-channel input, even for grayscale images.
    data = [(path, transform(Image.open(path).convert("RGB"))) for path in paths]

    # Separate paths and stack the image tensors into a single batch tensor.
    # This prepares them for batch processing by the model.
    return paths, torch.stack([img for _, img in data])

def get_top_k(query_embeds: torch.Tensor, gallery_embeds: torch.Tensor,
              gallery_paths: List[str], query_paths: List[str],
              k: int = 10, distance: str = "euclidean") -> dict:
    """
    Computes the top-k most similar gallery images for each query image based on their embeddings.

    This function calculates the distance (Euclidean or Cosine) between each
    query embedding and all gallery embeddings, then finds the `k` gallery
    images with the smallest distances (most similar).

    Args:
        query_embeds (torch.Tensor): A tensor of query image embeddings
                                     (shape: [num_queries, embedding_dim]).
        gallery_embeds (torch.Tensor): A tensor of gallery image embeddings
                                       (shape: [num_gallery_images, embedding_dim]).
        gallery_paths (List[str]): A list of full file paths for the gallery images,
                                   ordered to match `gallery_embeds`.
        query_paths (List[str]): A list of full file paths for the query images,
                                 ordered to match `query_embeds`.
        k (int, optional): The number of top similar images to retrieve for each query.
                           Defaults to 10.
        distance (str, optional): The distance metric to use. Can be "euclidean"
                                  (L2 distance) or "cosine" (1 - cosine similarity).
                                  Defaults to "euclidean".

    Returns:
        dict: A dictionary where keys are the base filenames of query images
              and values are lists of base filenames of their `k` most similar
              gallery images, sorted by similarity.
    """
    results = {} 

    # Iterate through each query embedding to find its similar gallery images.
    for i, query in enumerate(query_embeds):
        if distance == "cosine":
            # Calculate cosine distance (1 - cosine similarity)
            # query.unsqueeze(0) adds a batch dimension to allow broadcasting with gallery_embeds.
            dists = 1 - F.cosine_similarity(gallery_embeds, query.unsqueeze(0))
        elif distance == "euclidean":
            # Calculate Euclidean (L2) distance. Lower value means more similar.
            # torch.norm(A - B, dim=1) computes L2 norm along the embedding dimension for each row.
            dists = torch.norm(gallery_embeds - query, dim=1)
        else:
            # Raise an error if an unsupported distance metric is specified.
            raise ValueError(f"Unsupported distance metric: {distance}")

        # Find the indices of the `k` smallest distances (most similar images).
        topk = torch.topk(dists, k=k, largest=False)

        # Map the top-k indices back to the original gallery image filenames.
        # os.path.basename extracts just the filename from the full path.
        similar_images = [os.path.basename(gallery_paths[j]) for j in topk.indices]

        # Store the results in the dictionary, keyed by the query image's filename.
        results[os.path.basename(query_paths[i])] = similar_images

    return results

