import torch
from tqdm import tqdm

@torch.no_grad()
def extract_clip_embeddings(model, image_tensors, device, batch_size=32):
    """
    Extracts CLIP image embeddings for a given set of image tensors.

    This function processes image tensors in batches to efficiently extract
    features using a pre-trained CLIP model. The extracted features are then
    L2-normalized and concatenated to form a single tensor of embeddings.

    Args:
        model (CLIPModel): The pre-trained CLIP model. This model should have
                           an `encode_image` method.
        image_tensors (torch.Tensor): A tensor containing the preprocessed
                                      image data (e.g., shape: [N, C, H, W]).
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which
                               to perform the computations.
        batch_size (int, optional): The number of images to process in each
                                    batch. Defaults to 32.

    Returns:
        torch.Tensor: A tensor containing the L2-normalized CLIP embeddings
                      for all input images. The shape will be
                      [N, embedding_dim].
    """
    embeddings = []
    for i in tqdm(range(0, len(image_tensors), batch_size), desc="Extracting CLIP embeddings"):
        batch = image_tensors[i:i+batch_size].to(device)
        features = model.encode_image(batch)
        features /= features.norm(dim=-1, keepdim=True)  # normalize
        embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)
