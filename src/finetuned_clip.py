import torch
import torch.nn as nn
import clip

class FineTunedCLIP(nn.Module):
    """
    A PyTorch module that wraps a pre-trained CLIP model and adds a linear
    projection layer on top of its visual encoder.

    This class allows for freezing the original CLIP model's parameters
    and only training the added projection layer, or, if `freeze_clip` is
    set to False, allows the CLIP model itself to be fine-tuned
    (though the current implementation of `forward` and `encode_image`
    still uses `torch.no_grad()` for the CLIP part, which would prevent
    CLIP's parameters from being updated even if `freeze_clip` is False).
    """
    def __init__(self, device, embed_dim=128, freeze_clip=True):
        """
        Initializes the FineTunedCLIP model.

        Args:
            device (torch.device or str): The device (e.g., 'cuda', 'cpu')
                                          to load the CLIP model and projection
                                          layer onto.
            embed_dim (int, optional): The desired output embedding dimension
                                       after the projection layer. Defaults to 128.
            freeze_clip (bool, optional): If True, the parameters of the
                                          pre-trained CLIP model (both visual
                                          and text encoders) will be frozen,
                                          meaning only the `projection` layer
                                          will be trainable. Defaults to True.
        """
        super().__init__()
        self.device = device

        # Load the pre-trained CLIP model (ViT-B/32 architecture) and move it to the specified device.
        # The second return value (preprocess) is not needed here.
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.to(device)

        # Freeze the parameters of the loaded CLIP model if freeze_clip is True.
        # This prevents gradients from being computed for them during backpropagation.
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Define a linear projection layer that maps the output dimension of
        # CLIP's visual encoder to the desired `embed_dim`.
        # This layer is moved to the correct device immediately.
        self.projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, embed_dim)
        ).to(device)

        # Note: The `logit_scale` parameter is crucial for CLIP's contrastive
        # loss, but it's typically part of the CLIP model itself or managed
        # externally. If you intend to use this model with `CLIPLoss` (hybrid loss),
        # ensure `logit_scale` is accessible/trainable where needed (e.g., as
        # a `nn.Parameter` directly in this class if not part of `clip_model`).
        # The current clip.load returns a model that already has logit_scale.

    def forward(self, x):
        """
        Defines the forward pass of the model for image encoding.

        The original CLIP model's image encoding is performed without gradient
        tracking (`torch.no_grad()`), meaning its parameters are not updated
        through this path. The output of the CLIP visual encoder is then passed
        through the trainable projection layer.

        Args:
            x (torch.Tensor): Input image tensor (e.g., preprocessed batch of images).

        Returns:
            torch.Tensor: The projected image embedding.
        """
        with torch.no_grad(): # This ensures no gradients are computed for the self.clip_model part
            # Encode the input image tensor using CLIP's visual encoder.
            x = self.clip_model.encode_image(x.to(self.device))
        # Pass the CLIP features through the projection layer.
        # .float() ensures the input to nn.Linear is float32, which is standard.
        return self.projection(x.float())

    def encode_image(self, x):
        """
        Encodes an image tensor into a projected embedding.

        This method is identical to the `forward` method in this implementation.
        It's often provided as an explicit way to encode images, clearly separating
        it from a combined forward pass that might also handle text.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: The projected image embedding.
        """
        with torch.no_grad(): # This ensures no gradients are computed for the self.clip_model part
            x = self.clip_model.encode_image(x.to(self.device))
        return self.projection(x.float())

    # If you also need to encode text, you would add an `encode_text` method here:
    # def encode_text(self, text_tokens):
    #     """
    #     Encodes tokenized text into an embedding using the CLIP text encoder.
    #     Note: The `torch.no_grad()` might need to be adjusted if fine-tuning
    #     the CLIP text encoder is desired.
    #     """
    #     with torch.no_grad(): # Or without it, if fine-tuning text encoder
    #         text_features = self.clip_model.encode_text(text_tokens.to(self.device))
    #     # If you also want a projection for text features, add it here.
    #     # For simple CLIP fine-tuning, often the text features are used directly.
    #     return text_features.float()

    # The logit_scale parameter from the original CLIP model is typically accessed
    # for the hybrid loss. You might want to expose it directly if your
    # training loop needs it and it's not directly part of the model's `parameters()`
    # if `clip_model` is fully frozen. However, `clip.load` typically returns
    # a model where `model.logit_scale` is a `nn.Parameter`.
    @property
    def logit_scale(self):
        """
        Provides access to the logit_scale parameter from the underlying CLIP model.
        This parameter is crucial for the CLIP contrastive loss.
        """
        return self.clip_model.logit_scale