#!/usr/bin/env python3
"""
Interactive data visualization with matplotlib.
Shows data samples in a grid format.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from data.datasets import SRGANDataset, Noise2NoiseDataset, DeblurGANDataset
from torch.utils.data import DataLoader

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for matplotlib."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    # Denormalize
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    img = tensor[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def show_dataset_samples(dataset, dataloader, title, num_samples=6):
    """Display dataset samples in a grid."""
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        if len(batch) == 2:
            input_img, target_img = batch
            input_np = tensor_to_numpy(input_img)
            target_np = tensor_to_numpy(target_img)
            
            axes[0, i].imshow(input_np)
            axes[0, i].set_title(f'Input {i+1}', fontsize=9)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(target_np)
            axes[1, i].set_title(f'Target {i+1}', fontsize=9)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Main interactive visualization."""
    print("Loading datasets and generating visualizations...")
    
    # SRGAN
    print("\n1. Loading SRGAN data...")
    srgan_config = yaml.safe_load(open('configs/srgan.yaml'))
    srgan_dataset = SRGANDataset(
        data_dir=srgan_config['data']['data_dir'],
        hr_size=srgan_config['data']['hr_size'],
        lr_size=srgan_config['data']['lr_size'],
        num_channels=srgan_config['data']['num_channels'],
        normalize_mean=srgan_config['data']['normalize_mean'],
        normalize_std=srgan_config['data']['normalize_std'],
        is_train=True
    )
    srgan_loader = DataLoader(srgan_dataset, batch_size=1, shuffle=True)
    fig1 = show_dataset_samples(
        srgan_dataset, srgan_loader, 
        'SRGAN: Low-Resolution (top) → High-Resolution (bottom)', 
        num_samples=6
    )
    
    # Noise2Noise
    print("2. Loading Noise2Noise data...")
    n2n_config = yaml.safe_load(open('configs/noise2noise.yaml'))
    n2n_dataset = Noise2NoiseDataset(
        data_dir=n2n_config['data']['data_dir'],
        image_size=n2n_config['data']['image_size'],
        num_channels=n2n_config['data']['num_channels'],
        normalize_mean=n2n_config['data']['normalize_mean'],
        normalize_std=n2n_config['data']['normalize_std'],
        noise_type=n2n_config['data'].get('noise_type', 'gaussian'),
        noise_level=n2n_config['data'].get('noise_level', 0.1),
        is_train=True
    )
    n2n_loader = DataLoader(n2n_dataset, batch_size=1, shuffle=True)
    fig2 = show_dataset_samples(
        n2n_dataset, n2n_loader,
        f'Noise2Noise: Noisy (top) → Clean (bottom) [Noise: {n2n_config["data"].get("noise_type", "gaussian")}]',
        num_samples=6
    )
    
    # DeblurGAN
    print("3. Loading DeblurGAN data...")
    deblur_config = yaml.safe_load(open('configs/deblurgan.yaml'))
    deblur_dataset = DeblurGANDataset(
        data_dir=deblur_config['data']['data_dir'],
        image_size=deblur_config['data']['image_size'],
        num_channels=deblur_config['data']['num_channels'],
        normalize_mean=deblur_config['data']['normalize_mean'],
        normalize_std=deblur_config['data']['normalize_std'],
        blur_type=deblur_config['data'].get('blur_type', 'gaussian'),
        blur_size=deblur_config['data'].get('blur_size', 5),
        is_train=True
    )
    deblur_loader = DataLoader(deblur_dataset, batch_size=1, shuffle=True)
    fig3 = show_dataset_samples(
        deblur_dataset, deblur_loader,
        f'DeblurGAN: Blurred (top) → Sharp (bottom) [Blur: {deblur_config["data"].get("blur_type", "gaussian")}]',
        num_samples=6
    )
    
    print("\nDisplaying visualizations...")
    print("Close each window to see the next one.")
    
    plt.show()
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()

