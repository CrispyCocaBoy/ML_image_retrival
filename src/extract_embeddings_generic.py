# src/extract_embeddings_generic.py
import torch

@torch.no_grad()
def extract_embeddings(model, images, device):
    model = model.to(device)
    model.eval()
    images = images.to(device)
    return model(images)
