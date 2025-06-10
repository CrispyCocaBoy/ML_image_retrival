import torch.nn.functional as F
import torch

@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    model = model.to(device)
    model.eval()

    all_embeddings = [] 
    all_paths = []

    for images, paths in dataloader:
        images = images.to(device)
        embeddings = model(images)

        # 🔁 Normalize embeddings to unit vectors (L2 norm)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_paths.extend(paths)

    return torch.cat(all_embeddings, dim=0), all_paths