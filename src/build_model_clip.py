import torch
import clip  # Assicurati di aver eseguito: pip install git+https://github.com/openai/CLIP.git

def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess
