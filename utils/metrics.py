import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
import numpy as np
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, config: dict = None) -> float:
    """Calculate PSNR between two images."""
    # Denormalize if needed
    if config:
        mean = torch.tensor(config.get('data', {}).get('normalize_mean', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor(config.get('data', {}).get('normalize_std', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(img1.device)
        img1 = img1 * std + mean
        img2 = img2 * std + mean
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Convert to numpy
    if img1.dim() == 4:
        img1 = img1[0]
    if img2.dim() == 4:
        img2 = img2[0]
    
    img1_np = img1.cpu().permute(1, 2, 0).numpy()
    img2_np = img2.cpu().permute(1, 2, 0).numpy()
    
    # Calculate PSNR
    if img1_np.max() <= 1.0:
        img1_np = (img1_np * 255).astype(np.uint8)
        img2_np = (img2_np * 255).astype(np.uint8)
    
    return psnr_skimage(img1_np, img2_np, data_range=255)

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, config: dict = None) -> float:
    """Calculate SSIM between two images."""
    # Denormalize if needed
    if config:
        mean = torch.tensor(config.get('data', {}).get('normalize_mean', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor(config.get('data', {}).get('normalize_std', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(img1.device)
        img1 = img1 * std + mean
        img2 = img2 * std + mean
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Convert to numpy
    if img1.dim() == 4:
        img1 = img1[0]
    if img2.dim() == 4:
        img2 = img2[0]
    
    img1_np = img1.cpu().permute(1, 2, 0).numpy()
    img2_np = img2.cpu().permute(1, 2, 0).numpy()
    
    if img1_np.max() <= 1.0:
        img1_np = (img1_np * 255).astype(np.uint8)
        img2_np = (img2_np * 255).astype(np.uint8)
    
    # Calculate SSIM (multichannel for RGB)
    return ssim(img1_np, img2_np, data_range=255, multichannel=True, channel_axis=2)

_lpips_model = None

def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate LPIPS between two images."""
    global _lpips_model
    
    if not LPIPS_AVAILABLE:
        return 0.0
    
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(img1.device)
        _lpips_model.eval()
    
    # Ensure images are in [-1, 1] range for LPIPS
    img1_lpips = (img1 * 2.0) - 1.0
    img2_lpips = (img2 * 2.0) - 1.0
    
    with torch.no_grad():
        dist = _lpips_model(img1_lpips, img2_lpips)
    
    return dist.item()

def calculate_gradient_penalty(discriminator, real, fake, device):
    """Calculate gradient penalty for WGAN-GP."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

