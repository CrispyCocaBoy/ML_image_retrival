import os
import torch
import torch.nn.functional as F
from PIL import Image

def load_images_from_folder(folder, transform):
    paths = [os.path.join(folder, fname) for fname in os.listdir(folder)
             if fname.lower().endswith((".jpg", ".jpeg", ".png"))]
    data = [(path, transform(Image.open(path).convert("RGB"))) for path in paths]
    return paths, torch.stack([img for _, img in data])

def get_top_k(query_embeds, gallery_embeds, gallery_paths, query_paths, k=10, distance="l2"):
    results = {}

    for i, query in enumerate(query_embeds):
        if distance == "cosine":
            dists = 1 - F.cosine_similarity(gallery_embeds, query.unsqueeze(0))
        elif distance == "euclidean":
            dists = torch.norm(gallery_embeds - query, dim=1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")

        topk = torch.topk(dists, k=k, largest=False)
        similar_images = [os.path.basename(gallery_paths[j]) for j in topk.indices]
        results[os.path.basename(query_paths[i])] = similar_images

    return results
