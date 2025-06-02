import torch
import clip
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_clip_model(device):
    model, _ = clip.load("ViT-B/32", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    model.eval()
    return model, preprocess
