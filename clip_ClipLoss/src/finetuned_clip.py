import torch
import torch.nn as nn
import clip # Assuming clip library is installed and available

# Importing config for the dropout rate
from config import config

class FineTunedCLIP(nn.Module):
    """
    A PyTorch module that wraps a pre-trained CLIP model and adds linear
    projection layers with GELU activation and Dropout for both its visual
    and text encoders.

    This class allows for freezing the original CLIP model's parameters
    and only training the added projection layers, or, if `freeze_clip` is
    set to False, allows the CLIP model itself to be fine-tuned.
    
    Includes Dropout layers in the projection heads for regularization.
    """
    # --- MODIFIED SIGNATURE: Added 'clip_model=None' as a parameter ---
    def __init__(self, device, embed_dim=128, freeze_clip=True, clip_model=None):
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
            clip_model (torch.nn.Module, optional): An already loaded CLIP model instance.
                                                    If None, the model will be loaded internally.
        """
        super().__init__()
        self.device = device
        self.freeze_clip = freeze_clip 

        # --- MODIFIED LOGIC: Use passed clip_model or load it internally ---
        if clip_model is None:
            print("Loading CLIP model inside FineTunedCLIP. Consider passing it from main.py for efficiency.")
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
        else:
            # If clip_model is provided, ensure it's on the correct device
            self.clip_model = clip_model.to(device)

        if self.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.logit_scale.requires_grad = True # logit_scale should always be trainable

        # Define a linear projection layer for image features
        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, embed_dim),
            nn.GELU(), # <--- ADDED GELU ACTIVATION
            nn.Dropout(config.dropout_rate) 
        ).to(device)

        # Define a linear projection layer for text features
        self.text_projection = nn.Sequential(
            nn.Linear(512, embed_dim), # Fixed input dimension to 512 for CLIP ViT-B/32 text features
            nn.GELU(), # <--- ADDED GELU ACTIVATION
            nn.Dropout(config.dropout_rate) 
        ).to(device)
    
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
        clip_features = self.clip_model.encode_image(x.to(self.device))
        return self.image_projection(clip_features.float())

    def encode_text(self, text_tokens):
        """
        Encodes tokenized text into a projected embedding using the CLIP text encoder.
        The freezing of the CLIP backbone is managed by `requires_grad` flags set in `__init__`.
        """
        clip_features = self.clip_model.encode_text(text_tokens.to(self.device))
            
        return self.text_projection(clip_features.float())

    @property
    def logit_scale(self):
        """
        Provides access to the logit_scale parameter from the underlying CLIP model.
        """
        return self.clip_model.logit_scale
