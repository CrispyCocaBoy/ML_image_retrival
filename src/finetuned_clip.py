import torch
import torch.nn as nn
import clip # Assuming clip library is installed and available

# Importing config for the dropout rate
from config import config

class FineTunedCLIP(nn.Module):
    """
    A PyTorch module that wraps a pre-trained CLIP model and adds linear
    projection layers for both its visual and text encoders.

    This class allows for freezing the original CLIP model's parameters
    and only training the added projection layers, or, if `freeze_clip` is
    set to False, allows the CLIP model itself to be fine-tuned.
    
    Includes Dropout layers in the projection heads for regularization.
    """
    def __init__(self, device, embed_dim=128, freeze_clip=True):
        """
        Initializes the FineTunedCLIP model.

        Args:
            device (torch.device or str): The device (e.g., 'cuda', 'cpu')
                                          to load the CLIP model and projection
                                          layers onto.
            embed_dim (int, optional): The desired output embedding dimension
                                       for both image and text features after
                                       their respective projection layers. Defaults to 128.
            freeze_clip (bool, optional): If True, the parameters of the
                                          pre-trained CLIP model will be frozen.
                                          Only the `image_projection` and
                                          `text_projection` layers will be trainable.
                                          Defaults to True.
        """
        super().__init__()
        self.device = device
        self.freeze_clip = freeze_clip # Store this setting for potential future use or clarity

        # Load the pre-trained CLIP model (ViT-B/32 architecture) and move it to the specified device.
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        if self.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            # --- CRUCIAL FIX: Ensure logit_scale is always trainable ---
            # logit_scale is a critical learnable parameter in CLIP and must be updated
            # for the loss to be stable, even if the rest of the backbone is frozen.
            self.clip_model.logit_scale.requires_grad = True 

        # Define a linear projection layer for image features, now including Dropout.
        # Dropout is added here for regularization on the new projection layer.
        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, embed_dim),
            nn.Dropout(config.dropout_rate) # Add Dropout layer
        ).to(device)

        # Define a linear projection layer for text features, now including Dropout.
        # Dropout is added here for regularization on the new projection layer.
        self.text_projection = nn.Sequential(
            nn.Linear(512, embed_dim), # Fixed input dimension to 512 for CLIP ViT-B/32 text features
            nn.Dropout(config.dropout_rate) # Add Dropout layer
        ).to(device)

    # --- SIMPLIFIED: Removed `with torch.no_grad()` from `encode_image` and `encode_text`. ---
    # The `requires_grad=False` setting on parameters already handles freezing the backbone.
    # This ensures that `logit_scale` (which has `requires_grad=True`) can receive gradients.
    
    def forward(self, x):
        """
        Defines the forward pass of the model for image encoding.
        For training with CLIP, `encode_image` and `encode_text` are typically called directly.
        """
        # Call encode_image directly; autograd will handle gradient tracking based on requires_grad flags.
        return self.image_projection(self.clip_model.encode_image(x.to(self.device)).float())

    def encode_image(self, x):
        """
        Encodes an image tensor into a projected embedding.
        The freezing of the CLIP backbone is managed by `requires_grad` flags set in `__init__`.
        """
        # Ensure input is on the correct device. Autograd handles gradient flow.
        clip_features = self.clip_model.encode_image(x.to(self.device))
        return self.image_projection(clip_features.float())

    def encode_text(self, text_tokens):
        """
        Encodes tokenized text into a projected embedding using the CLIP text encoder.
        The freezing of the CLIP backbone is managed by `requires_grad` flags set in `__init__`.
        """
        # Ensure input is on the correct device. Autograd handles gradient flow.
        clip_features = self.clip_model.encode_text(text_tokens.to(self.device))
            
        return self.text_projection(clip_features.float())

    @property
    def logit_scale(self):
        """
        Provides access to the logit_scale parameter from the underlying CLIP model.
        """
        return self.clip_model.logit_scale
