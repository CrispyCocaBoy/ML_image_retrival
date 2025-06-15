import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_distance(emb1, emb2):
    if emb1.shape != emb2.shape:
        raise ValueError("emb1 e emb2 devono avere la stessa forma")
    
    else:
        # 1 - cosine similarity = cosine distance
        return 1 - F.cosine_similarity(emb1, emb2, dim=1)


