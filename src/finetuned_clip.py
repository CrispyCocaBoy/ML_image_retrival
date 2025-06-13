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

        # Load the pre-trained CLIP model (ViT-B/32 architecture) and move it to the specified device.
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

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

    # Note: The forward, encode_image, and encode_text methods still use
    # `torch.no_grad()` for the CLIP backbone. This is correct if `freeze_clip` is True.
    # If `freeze_clip` were False, you would remove `torch.no_grad()` from these methods
    # to allow the backbone to train.
    
    def forward(self, x):
        """
        Defines the forward pass of the model for image encoding.
        This method should typically not be called directly for training
        as it includes no_grad(). `encode_image` and `encode_text` are used for training.
        """
        # It's typical for the main CLIP backbone to be in no_grad if frozen
        with torch.no_grad(): 
            x = self.clip_model.encode_image(x.to(self.device))
        return self.image_projection(x.float())

    def encode_image(self, x):
        """
        Encodes an image tensor into a projected embedding.
        If `freeze_clip` is True, this runs the CLIP backbone in no_grad.
        """
        # Only detach gradients from CLIP backbone if it's frozen
        if not self.clip_model.training: # Check if model is in eval mode (like for inference)
             with torch.no_grad():
                 x = self.clip_model.encode_image(x.to(self.device))
        else: # During training, if CLIP backbone is unfrozen, gradients flow
            # If freeze_clip was true, then parameters are already set to requires_grad=False
            # so no_grad is fine. If freeze_clip is false, you generally remove the no_grad()
            # block here to allow gradients to flow through the backbone.
            # However, for this specific setup where freeze_clip is True, the `with torch.no_grad():` is fine.
            x = self.clip_model.encode_image(x.to(self.device))

        return self.image_projection(x.float())

    def encode_text(self, text_tokens):
        """
        Encodes tokenized text into a projected embedding using the CLIP text encoder.
        If `freeze_clip` is True, this runs the CLIP backbone in no_grad.
        """
        # Only detach gradients from CLIP backbone if it's frozen
        if not self.clip_model.training: # Check if model is in eval mode (like for inference)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens.to(self.device))
        else: # During training, if CLIP backbone is unfrozen, gradients flow
            # If freeze_clip was true, then parameters are already set to requires_grad=False
            # so no_grad is fine. If freeze_clip is false, you generally remove the no_grad()
            # block here to allow gradients to flow through the backbone.
            text_features = self.clip_model.encode_text(text_tokens.to(self.device))
            
        return self.text_projection(text_features.float()) # Apply text_projection here!

    @property
    def logit_scale(self):
        """
        Provides access to the logit_scale parameter from the underlying CLIP model.
        """
        return self.clip_model.logit_scale
