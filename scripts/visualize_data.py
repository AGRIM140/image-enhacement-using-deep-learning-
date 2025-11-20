#!/usr/bin/env python3
"""
Script to visualize training data for all models.
Shows what the models will be training on.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from data.datasets import SRGANDataset, Noise2NoiseDataset, DeblurGANDataset
from torch.utils.data import DataLoader

def denormalize_tensor(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize tensor for visualization."""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for matplotlib."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def visualize_srgan_data(num_samples=4):
    """Visualize SRGAN training data (LR/HR pairs)."""
    print("Loading SRGAN dataset...")
    config = yaml.safe_load(open('configs/srgan.yaml'))
    
    dataset = SRGANDataset(
        data_dir=config['data']['data_dir'],
        hr_size=config['data']['hr_size'],
        lr_size=config['data']['lr_size'],
        num_channels=config['data']['num_channels'],
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std'],
        is_train=True
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    fig.suptitle('SRGAN Training Data (Low-Resolution → High-Resolution)', fontsize=16, fontweight='bold')
    
    for i, (lr_img, hr_img) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        lr_np = tensor_to_numpy(lr_img)
        hr_np = tensor_to_numpy(hr_img)
        
        axes[i, 0].imshow(lr_np)
        axes[i, 0].set_title(f'Sample {i+1}: Low-Resolution ({config["data"]["lr_size"]}x{config["data"]["lr_size"]})', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(hr_np)
        axes[i, 1].set_title(f'Sample {i+1}: High-Resolution ({config["data"]["hr_size"]}x{config["data"]["hr_size"]})', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_visualization_srgan.png', dpi=150, bbox_inches='tight')
    print("Saved: data_visualization_srgan.png")
    return fig

def visualize_noise2noise_data(num_samples=4):
    """Visualize Noise2Noise training data (noisy/clean pairs)."""
    print("Loading Noise2Noise dataset...")
    config = yaml.safe_load(open('configs/noise2noise.yaml'))
    
    dataset = Noise2NoiseDataset(
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        num_channels=config['data']['num_channels'],
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std'],
        noise_type=config['data'].get('noise_type', 'gaussian'),
        noise_level=config['data'].get('noise_level', 0.1),
        is_train=True
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    fig.suptitle(f'Noise2Noise Training Data (Noisy → Clean)\nNoise Type: {config["data"].get("noise_type", "gaussian")}, Level: {config["data"].get("noise_level", 0.1)}', 
                 fontsize=16, fontweight='bold')
    
    for i, (noisy_img, clean_img) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        noisy_np = tensor_to_numpy(noisy_img)
        clean_np = tensor_to_numpy(clean_img)
        
        axes[i, 0].imshow(noisy_np)
        axes[i, 0].set_title(f'Sample {i+1}: Noisy Input', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(clean_np)
        axes[i, 1].set_title(f'Sample {i+1}: Clean Target', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_visualization_noise2noise.png', dpi=150, bbox_inches='tight')
    print("Saved: data_visualization_noise2noise.png")
    return fig

def visualize_deblurgan_data(num_samples=4):
    """Visualize DeblurGAN training data (blurred/sharp pairs)."""
    print("Loading DeblurGAN dataset...")
    config = yaml.safe_load(open('configs/deblurgan.yaml'))
    
    dataset = DeblurGANDataset(
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        num_channels=config['data']['num_channels'],
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std'],
        blur_type=config['data'].get('blur_type', 'gaussian'),
        blur_size=config['data'].get('blur_size', 5),
        is_train=True
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    fig.suptitle(f'DeblurGAN Training Data (Blurred → Sharp)\nBlur Type: {config["data"].get("blur_type", "gaussian")}, Size: {config["data"].get("blur_size", 5)}', 
                 fontsize=16, fontweight='bold')
    
    for i, (blurred_img, sharp_img) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        blurred_np = tensor_to_numpy(blurred_img)
        sharp_np = tensor_to_numpy(sharp_img)
        
        axes[i, 0].imshow(blurred_np)
        axes[i, 0].set_title(f'Sample {i+1}: Blurred Input', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sharp_np)
        axes[i, 1].set_title(f'Sample {i+1}: Sharp Target', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_visualization_deblurgan.png', dpi=150, bbox_inches='tight')
    print("Saved: data_visualization_deblurgan.png")
    return fig

def check_data_status():
    """Check if real data exists or synthetic data is being used."""
    print("\n" + "="*60)
    print("DATA STATUS CHECK")
    print("="*60)
    
    configs = [
        ('configs/srgan.yaml', 'data/srgan', 'SRGAN'),
        ('configs/noise2noise.yaml', 'data/noise2noise', 'Noise2Noise'),
        ('configs/deblurgan.yaml', 'data/deblurgan', 'DeblurGAN'),
    ]
    
    for config_path, data_dir, model_name in configs:
        data_path = Path(data_dir)
        images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png")) + \
                 list(data_path.glob("*.JPG")) + list(data_path.glob("*.PNG"))
        
        if len(images) > 0:
            print(f"\n[{model_name}]")
            print(f"  [REAL DATA] Found {len(images)} images in {data_dir}")
        else:
            print(f"\n[{model_name}]")
            print(f"  [SYNTHETIC DATA] No images found in {data_dir}")
            print(f"  Using automatically generated synthetic data")
    
    print("\n" + "="*60)

def main():
    """Main visualization function."""
    print("="*60)
    print("DATA VISUALIZATION")
    print("="*60)
    print()
    
    # Check data status
    check_data_status()
    
    print("\nGenerating visualizations...")
    print("(This may take a moment to generate synthetic data samples)\n")
    
    try:
        # Visualize each dataset
        visualize_srgan_data(num_samples=4)
        visualize_noise2noise_data(num_samples=4)
        visualize_deblurgan_data(num_samples=4)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - data_visualization_srgan.png")
        print("  - data_visualization_noise2noise.png")
        print("  - data_visualization_deblurgan.png")
        print("\nThese images show what your models are training on.")
        print("Open them to see sample input/target pairs for each model.")
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

