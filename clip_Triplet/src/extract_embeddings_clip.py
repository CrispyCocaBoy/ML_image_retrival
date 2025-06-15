import torch
from tqdm import tqdm

@torch.no_grad()
def extract_clip_embeddings(model, image_tensors, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(image_tensors), batch_size), desc="Extracting CLIP embeddings"):
        batch = image_tensors[i:i+batch_size].to(device)
        features = model.encode_image(batch)
        features /= features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)
