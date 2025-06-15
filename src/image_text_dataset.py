import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import clip # Needed for text tokenization

class ImageTextDataset(Dataset):
    """
    Dataset for loading image-text pairs from a structured folder.

    Assumes `data_dir` contains subfolders named by class (e.g., '1', '2', '3'),
    each containing images. Text descriptions are generated based on these class names.
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.text_descriptions = [] # Stores raw text descriptions

        # Load CLIP's tokenizer
        _, _ = clip.load("ViT-B/32", device="cpu")
        self.tokenizer = clip.tokenize

        # Iterate through class folders (e.g., '1', '2', '3')
        for class_name in sorted(os.listdir(data_dir)):
            class_folder_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder_path):
                try:
                    num_class = int(class_name)
                    # Convert number to word, or map to a specific label if you have one
                    text_desc = f"a photo of class number {num_class}" # Generic fallback
                except ValueError:
                    # If class_name is not a number 
                    text_desc = f"a photo of a {class_name}"


                # List all images within this class folder
                for img_filename in os.listdir(class_folder_path):
                    if img_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_folder_path, img_filename))
                        self.text_descriptions.append(text_desc) # Assign the description to the image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Retrieves an image and its tokenized text description.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - image (torch.Tensor): The transformed image tensor.
                - text_tokens (torch.Tensor): The tokenized text description tensor.
        """
        img_path = self.image_paths[idx]
        text_desc = self.text_descriptions[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize text description using CLIP's tokenizer
        text_tokens = self.tokenizer(text_desc).squeeze(0) # squeeze(0) removes the batch dimension added by tokenizer

        return image, text_tokens
