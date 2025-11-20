import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def load_image(path: str, config: dict = None) -> torch.Tensor:
    """Load and preprocess an image."""
    img = Image.open(path).convert('RGB')
    
    if config:
        mean = config.get('data', {}).get('normalize_mean', [0.5, 0.5, 0.5])
        std = config.get('data', {}).get('normalize_std', [0.5, 0.5, 0.5])
        image_size = config.get('data', {}).get('image_size', 256)
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image_size = 256
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform(img).unsqueeze(0)

def save_image(tensor: torch.Tensor, path: str, config: dict = None):
    """Save a tensor as an image."""
    if config:
        mean = config.get('data', {}).get('normalize_mean', [0.5, 0.5, 0.5])
        std = config.get('data', {}).get('normalize_std', [0.5, 0.5, 0.5])
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    # Denormalize
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL and save
    if tensor.dim() == 4:
        tensor = tensor[0]
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(path)

