import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import random
import cv2

class SRGANDataset(Dataset):
    def __init__(self, data_dir, hr_size=256, lr_size=64, num_channels=3, 
                 normalize_mean=[0.5, 0.5, 0.5], normalize_std=[0.5, 0.5, 0.5],
                 use_color_jitter=True, is_train=True):
        self.data_dir = Path(data_dir)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.is_train = is_train
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            # If no images found, create synthetic data
            print(f"Warning: No images found in {data_dir}. Will use synthetic data.")
            self.image_paths = None
        
        # HR transforms
        hr_transforms = []
        if is_train:
            hr_transforms.append(transforms.RandomCrop(hr_size))
            if use_color_jitter:
                hr_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        else:
            hr_transforms.append(transforms.CenterCrop(hr_size))
        hr_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        self.hr_transform = transforms.Compose(hr_transforms)
        
        # LR transforms (downsample HR)
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    
    def __len__(self):
        if self.image_paths is None:
            return 1000  # Synthetic dataset size
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.image_paths is None or idx >= len(self.image_paths):
            # Generate synthetic data
            hr_img = torch.randn(3, self.hr_size, self.hr_size)
            lr_img = torch.nn.functional.interpolate(
                hr_img.unsqueeze(0), size=(self.lr_size, self.lr_size), mode='bilinear', align_corners=False
            ).squeeze(0)
            return lr_img, hr_img
        
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)
        
        return lr_img, hr_img

class Noise2NoiseDataset(Dataset):
    def __init__(self, data_dir, image_size=256, num_channels=3,
                 normalize_mean=[0.5, 0.5, 0.5], normalize_std=[0.5, 0.5, 0.5],
                 noise_type='gaussian', noise_level=0.1, use_color_jitter=True, is_train=True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.is_train = is_train
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}. Will use synthetic data.")
            self.image_paths = None
        
        # Base transforms
        base_transforms = []
        if is_train:
            base_transforms.append(transforms.RandomCrop(image_size))
            if use_color_jitter:
                base_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        else:
            base_transforms.append(transforms.CenterCrop(image_size))
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        self.transform = transforms.Compose(base_transforms)
    
    def __len__(self):
        if self.image_paths is None:
            return 1000
        return len(self.image_paths)
    
    def add_noise(self, img_tensor):
        """Add noise to image tensor."""
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img_tensor) * self.noise_level
            noisy = img_tensor + noise
        elif self.noise_type == 'salt_pepper':
            noisy = img_tensor.clone()
            mask = torch.rand_like(img_tensor) < self.noise_level
            noisy[mask] = torch.rand_like(noisy[mask])
        else:  # poisson
            noisy = torch.poisson(img_tensor * 10) / 10.0
        
        return torch.clamp(noisy, -1, 1)
    
    def __getitem__(self, idx):
        if self.image_paths is None or idx >= len(self.image_paths):
            # Generate synthetic data
            clean_img = torch.randn(3, self.image_size, self.image_size)
            noisy_img = self.add_noise(clean_img)
            return noisy_img, clean_img
        
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        clean_img = self.transform(img)
        noisy_img = self.add_noise(clean_img)
        
        return noisy_img, clean_img

class DeblurGANDataset(Dataset):
    def __init__(self, data_dir, image_size=256, num_channels=3,
                 normalize_mean=[0.5, 0.5, 0.5], normalize_std=[0.5, 0.5, 0.5],
                 blur_type='gaussian', blur_size=5, use_color_jitter=True, is_train=True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.blur_type = blur_type
        self.blur_size = blur_size
        self.is_train = is_train
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}. Will use synthetic data.")
            self.image_paths = None
        
        # Base transforms
        base_transforms = []
        if is_train:
            base_transforms.append(transforms.RandomCrop(image_size))
            if use_color_jitter:
                base_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        else:
            base_transforms.append(transforms.CenterCrop(image_size))
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        self.transform = transforms.Compose(base_transforms)
    
    def __len__(self):
        if self.image_paths is None:
            return 1000
        return len(self.image_paths)
    
    def apply_blur(self, img_tensor):
        """Apply blur to image tensor."""
        # Convert to numpy for blur
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np + 1) / 2.0  # Denormalize to [0, 1]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Apply blur
        if self.blur_type == 'gaussian':
            blurred = cv2.GaussianBlur(img_np, (self.blur_size, self.blur_size), 0)
        elif self.blur_type == 'motion':
            kernel = np.ones((self.blur_size, 1), np.float32) / self.blur_size
            blurred = cv2.filter2D(img_np, -1, kernel)
        else:  # defocus
            blurred = cv2.GaussianBlur(img_np, (self.blur_size, self.blur_size), 0)
        
        # Convert back to tensor
        blurred = blurred.astype(np.float32) / 255.0
        blurred = (blurred * 2.0) - 1.0  # Normalize to [-1, 1]
        blurred = torch.from_numpy(blurred).permute(2, 0, 1)
        return blurred
    
    def __getitem__(self, idx):
        if self.image_paths is None or idx >= len(self.image_paths):
            # Generate synthetic data
            sharp_img = torch.randn(3, self.image_size, self.image_size)
            blurred_img = self.apply_blur(sharp_img)
            return blurred_img, sharp_img
        
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        sharp_img = self.transform(img)
        blurred_img = self.apply_blur(sharp_img)
        
        return blurred_img, sharp_img

def get_dataloader(config, model_type, is_train=True):
    """Get appropriate dataloader based on model type."""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    num_workers = data_cfg.get('num_workers', 4)
    
    if model_type == 'srgan':
        dataset = SRGANDataset(
            data_dir=data_cfg['data_dir'],
            hr_size=data_cfg.get('hr_size', 256),
            lr_size=data_cfg.get('lr_size', 64),
            num_channels=data_cfg.get('num_channels', 3),
            normalize_mean=data_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
            normalize_std=data_cfg.get('normalize_std', [0.5, 0.5, 0.5]),
            use_color_jitter=data_cfg.get('use_color_jitter', True),
            is_train=is_train
        )
    elif model_type == 'noise2noise':
        dataset = Noise2NoiseDataset(
            data_dir=data_cfg['data_dir'],
            image_size=data_cfg.get('image_size', 256),
            num_channels=data_cfg.get('num_channels', 3),
            normalize_mean=data_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
            normalize_std=data_cfg.get('normalize_std', [0.5, 0.5, 0.5]),
            noise_type=data_cfg.get('noise_type', 'gaussian'),
            noise_level=data_cfg.get('noise_level', 0.1),
            use_color_jitter=data_cfg.get('use_color_jitter', True),
            is_train=is_train
        )
    elif model_type == 'deblurgan':
        dataset = DeblurGANDataset(
            data_dir=data_cfg['data_dir'],
            image_size=data_cfg.get('image_size', 256),
            num_channels=data_cfg.get('num_channels', 3),
            normalize_mean=data_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
            normalize_std=data_cfg.get('normalize_std', [0.5, 0.5, 0.5]),
            blur_type=data_cfg.get('blur_type', 'gaussian'),
            blur_size=data_cfg.get('blur_size', 5),
            use_color_jitter=data_cfg.get('use_color_jitter', True),
            is_train=is_train
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True
    )

