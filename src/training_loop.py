import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from src.finetuned_clip import FineTunedCLIP

class CLIPLoss(nn.Module):
    """
    Implements the Contrastive Language-Image Pre-training (CLIP) loss.

    This loss function calculates symmetric cross-entropy between image and text
    embeddings based on their cosine similarity. It requires a learnable
    temperature parameter (logit_scale) from the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # Normalize features to prepare for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between all image and text features in the batch.
        # The result is a square matrix where element (i, j) represents the similarity
        # between the i-th image embedding and the j-th text embedding.
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # Create labels for cross-entropy: the diagonal elements correspond to the
        # correct (positive) pairs. For a batch of N items, the labels are [0, 1, ..., N-1].
        labels = torch.arange(len(image_features), device=image_features.device)

        # Calculate the total loss as the average of two cross-entropy losses:
        # 1. Image-to-text matching (predicting which text matches each image)
        # 2. Text-to-image matching (predicting which image matches each text)
        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss

def train_clip_hybrid(model, dataloader, device="cuda", epochs=10, lr=1e-5):
    """
    Trains a given PyTorch model using the CLIP-style hybrid (contrastive + cross-entropy) loss.

    This function sets up the model for training, defines an Adam optimizer
    and a CLIPLoss criterion. It then iterates through the specified number
    of epochs, processing image-text pairs from the dataloader. For each batch,
    it encodes both images and texts, computes the CLIP loss, performs
    backpropagation, and updates the model's weights and the learnable temperature.
    Training progress and loss are displayed using tqdm.

    Args:
        model (torch.nn.Module): The neural network model to be trained (e.g., FineTunedCLIP).
                                 It should have methods `encode_image` and `encode_text`
                                 and expose a `logit_scale` parameter (e.g., via a property).
        dataloader (torch.utils.data.DataLoader): A dataloader that yields
                                                   batches of (image_tensor, text_token_tensor) pairs.
        device (str, optional): The device ('cuda' or 'cpu') to run the
                                training on. Defaults to "cuda".
        epochs (int, optional): The number of training epochs. Defaults to 10.
        lr (float, optional): The learning rate for the Adam optimizer.
                              Defaults to 1e-5.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)
    # Initialize the Adam optimizer with the model's parameters.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Initialize the CLIPLoss criterion.
    criterion = CLIPLoss()

    # Set the model to training mode. This affects behavior of modules like BatchNorm and Dropout.
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Wrap the dataloader with tqdm for a progress bar during training.
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, texts in loop:
            # Move image and text tensors to the specified device.
            images = images.to(device)
            texts = texts.to(device) # texts should already be tokenized tensors from the dataset

            # Zero the gradients of the optimizer before each backward pass.
            optimizer.zero_grad()

            # Encode images and texts using the model's respective encoders.
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Calculate the CLIP loss using the encoded features and the model's logit_scale.
            loss = criterion(image_features, text_features, model.logit_scale)
            # Perform backpropagation to compute gradients.
            loss.backward()
            # Update model parameters using the optimizer.
            optimizer.step()

            # Accumulate total loss and update the progress bar.
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Print average loss for the epoch.
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model
