#!/usr/bin/env python3
"""
Visualize training metrics and model output quality.
Shows training curves, validation metrics, and sample outputs.
"""
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from tools.model_runner import ModelRunner
from utils.image_utils import load_image, save_image
from utils import metrics
import torchvision.transforms as transforms
from PIL import Image

def load_metrics_from_json(metrics_file):
    """Load metrics from JSON file."""
    if not Path(metrics_file).exists():
        return None
    try:
        with open(metrics_file) as f:
            return json.load(f)
    except:
        return None

def parse_log_file(log_file):
    """Parse training log file to extract metrics."""
    if not Path(log_file).exists():
        return None
    
    metrics_data = {
        'epochs': [],
        'train_g_loss': [],
        'train_d_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_lpips': []
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Look for epoch information
                if 'Epoch' in line:
                    # Extract epoch number
                    epoch_match = re.search(r'Epoch (\d+)', line)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        metrics_data['epochs'].append(epoch)
                    
                    # Extract metrics
                    if 'g_loss' in line.lower():
                        g_loss_match = re.search(r'g_loss[:\s]+([\d.]+)', line, re.IGNORECASE)
                        if g_loss_match:
                            metrics_data['train_g_loss'].append(float(g_loss_match.group(1)))
                    
                    if 'd_loss' in line.lower():
                        d_loss_match = re.search(r'd_loss[:\s]+([\d.]+)', line, re.IGNORECASE)
                        if d_loss_match:
                            metrics_data['train_d_loss'].append(float(d_loss_match.group(1)))
                    
                    if 'psnr' in line.lower():
                        psnr_match = re.search(r'psnr[:\s]+([\d.]+)', line, re.IGNORECASE)
                        if psnr_match:
                            metrics_data['val_psnr'].append(float(psnr_match.group(1)))
                    
                    if 'ssim' in line.lower():
                        ssim_match = re.search(r'ssim[:\s]+([\d.]+)', line, re.IGNORECASE)
                        if ssim_match:
                            metrics_data['val_ssim'].append(float(ssim_match.group(1)))
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None
    
    return metrics_data if metrics_data['epochs'] else None

def plot_training_curves(model_name, metrics_data, log_data):
    """Plot training curves for losses and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Training Metrics', fontsize=16, fontweight='bold')
    
    # Get data from JSON metrics or log file
    epochs = []
    train_g_loss = []
    train_d_loss = []
    val_psnr = []
    val_ssim = []
    
    if metrics_data:
        # Extract from JSON
        for key, values in metrics_data.items():
            if 'train_g_loss' in key or 'train_g_loss' == key:
                epochs = [int(k.split('_')[0]) for k in metrics_data.keys() if k.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
                break
    
    if log_data and log_data['epochs']:
        epochs = log_data['epochs']
        train_g_loss = log_data['train_g_loss'][:len(epochs)]
        train_d_loss = log_data['train_d_loss'][:len(epochs)]
        val_psnr = log_data['val_psnr'][:len(epochs)]
        val_ssim = log_data['val_ssim'][:len(epochs)]
    
    # Plot Generator Loss
    if train_g_loss:
        axes[0, 0].plot(epochs[:len(train_g_loss)], train_g_loss, 'b-', linewidth=2, label='Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Generator Loss')
    
    # Plot Discriminator Loss
    if train_d_loss:
        axes[0, 1].plot(epochs[:len(train_d_loss)], train_d_loss, 'r-', linewidth=2, label='Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Discriminator Loss')
    
    # Plot PSNR
    if val_psnr:
        axes[1, 0].plot(epochs[:len(val_psnr)], val_psnr, 'g-', linewidth=2, label='PSNR')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('Validation PSNR (Higher is Better)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        if val_psnr:
            axes[1, 0].axhline(y=max(val_psnr), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(val_psnr):.2f}')
    else:
        axes[1, 0].text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Validation PSNR')
    
    # Plot SSIM
    if val_ssim:
        axes[1, 1].plot(epochs[:len(val_ssim)], val_ssim, 'm-', linewidth=2, label='SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('Validation SSIM (Higher is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        if val_ssim:
            axes[1, 1].axhline(y=max(val_ssim), color='m', linestyle='--', alpha=0.5, label=f'Best: {max(val_ssim):.4f}')
    else:
        axes[1, 1].text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Validation SSIM')
    
    plt.tight_layout()
    return fig

def generate_test_sample(model_type, config_path, checkpoint_path=None):
    """Generate a test sample using the model."""
    try:
        config = yaml.safe_load(open(config_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a synthetic test image
        if model_type == 'srgan':
            # Low-res test image
            test_img = torch.randn(1, 3, config['data']['lr_size'], config['data']['lr_size'])
        else:
            # Regular test image
            test_img = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size'])
        
        test_img = test_img.to(device)
        
        # Load model if checkpoint exists
        if checkpoint_path and Path(checkpoint_path).exists():
            runner = ModelRunner(model_type, config, device)
            runner.load_checkpoint(checkpoint_path)
            runner.generator.eval()
            
            with torch.no_grad():
                output = runner.generator(test_img)
            
            return test_img.cpu(), output.cpu()
        else:
            # Return input only if no checkpoint
            return test_img.cpu(), None
    except Exception as e:
        print(f"Error generating test sample: {e}")
        return None, None

def visualize_model_outputs(model_name, model_type, config_path, checkpoint_path=None):
    """Visualize model input/output pairs."""
    input_img, output_img = generate_test_sample(model_type, config_path, checkpoint_path)
    
    if input_img is None:
        return None
    
    fig, axes = plt.subplots(1, 2 if output_img is not None else 1, figsize=(12, 6))
    if output_img is None:
        axes = [axes]
    
    fig.suptitle(f'{model_name} - Model Output', fontsize=14, fontweight='bold')
    
    # Denormalize and convert to numpy
    def tensor_to_numpy(t):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        t = t * std + mean
        t = torch.clamp(t, 0, 1)
        if t.dim() == 4:
            t = t[0]
        return t.permute(1, 2, 0).cpu().numpy()
    
    input_np = tensor_to_numpy(input_img)
    axes[0].imshow(input_np)
    axes[0].set_title('Input', fontsize=12)
    axes[0].axis('off')
    
    if output_img is not None:
        output_np = tensor_to_numpy(output_img)
        axes[1].imshow(output_np)
        axes[1].set_title('Model Output', fontsize=12)
        axes[1].axis('off')
    else:
        axes[0].text(0.5, -0.1, 'No checkpoint available - showing input only', 
                    ha='center', transform=axes[0].transAxes, fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def calculate_model_accuracy_metrics(model_type, config_path, checkpoint_path=None, num_samples=5):
    """Calculate accuracy metrics on test samples."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    
    try:
        config = yaml.safe_load(open(config_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runner = ModelRunner(model_type, config, device)
        runner.load_checkpoint(checkpoint_path)
        runner.generator.eval()
        
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        
        for _ in range(num_samples):
            # Generate synthetic input/target pair
            if model_type == 'srgan':
                target = torch.randn(1, 3, config['data']['hr_size'], config['data']['hr_size']).to(device)
                input_img = torch.nn.functional.interpolate(
                    target, size=(config['data']['lr_size'], config['data']['lr_size']), 
                    mode='bilinear', align_corners=False
                )
            elif model_type == 'noise2noise':
                target = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size']).to(device)
                noise = torch.randn_like(target) * config['data'].get('noise_level', 0.1)
                input_img = torch.clamp(target + noise, -1, 1)
            else:  # deblurgan
                target = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size']).to(device)
                # Simple blur simulation
                input_img = torch.nn.functional.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
            
            with torch.no_grad():
                output = runner.generator(input_img)
            
            # Calculate metrics
            psnr = metrics.calculate_psnr(output, target, config)
            ssim = metrics.calculate_ssim(output, target, config)
            lpips = metrics.calculate_lpips(output, target)
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            lpips_scores.append(lpips)
        
        return {
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'lpips_mean': np.mean(lpips_scores),
            'lpips_std': np.std(lpips_scores)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def plot_accuracy_summary(models_metrics):
    """Plot accuracy summary for all models."""
    if not any(models_metrics.values()):
        print("No metrics available to plot")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Accuracy Summary', fontsize=16, fontweight='bold')
    
    model_names = []
    psnr_means = []
    ssim_means = []
    lpips_means = []
    
    for model_name, metrics in models_metrics.items():
        if metrics:
            model_names.append(model_name)
            psnr_means.append(metrics['psnr_mean'])
            ssim_means.append(metrics['ssim_mean'])
            lpips_means.append(metrics['lpips_mean'])
    
    if model_names:
        x = np.arange(len(model_names))
        width = 0.6
        
        # PSNR
        axes[0].bar(x, psnr_means, width, label='PSNR', color='green', alpha=0.7)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('Peak Signal-to-Noise Ratio\n(Higher is Better)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=15, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # SSIM
        axes[1].bar(x, ssim_means, width, label='SSIM', color='blue', alpha=0.7)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('Structural Similarity Index\n(Higher is Better)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=15, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # LPIPS
        axes[2].bar(x, lpips_means, width, label='LPIPS', color='red', alpha=0.7)
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('LPIPS')
        axes[2].set_title('Learned Perceptual Image Patch Similarity\n(Lower is Better)')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names, rotation=15, ha='right')
        axes[2].grid(True, alpha=0.3, axis='y')
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def main():
    """Main visualization function."""
    print("="*60)
    print("METRICS AND ACCURACY VISUALIZATION")
    print("="*60)
    print()
    
    models = [
        ('SRGAN', 'srgan', 'configs/srgan.yaml', 'checkpoints/srgan_model/best.pth'),
        ('Noise2Noise', 'noise2noise', 'configs/noise2noise.yaml', 'checkpoints/noise2noise_model/best.pth'),
        ('DeblurGAN', 'deblurgan', 'configs/deblurgan.yaml', 'checkpoints/deblurgan_model/best.pth'),
    ]
    
    all_metrics = {}
    
    for model_name, model_type, config_path, checkpoint_path in models:
        print(f"\nProcessing {model_name}...")
        
        # Load metrics
        metrics_file = Path(checkpoint_path).parent / 'metrics.json'
        metrics_data = load_metrics_from_json(metrics_file)
        
        # Parse log file
        log_dir = Path('logs') / f'{model_type}_model'
        log_files = list(log_dir.glob('*.log'))
        log_data = None
        if log_files:
            log_data = parse_log_file(max(log_files, key=lambda x: x.stat().st_mtime))
        
        # Plot training curves
        fig1 = plot_training_curves(model_name, metrics_data, log_data)
        if fig1:
            fig1.savefig(f'metrics_training_{model_type}.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: metrics_training_{model_type}.png")
        
        # Visualize model outputs
        fig2 = visualize_model_outputs(model_name, model_type, config_path, checkpoint_path)
        if fig2:
            fig2.savefig(f'metrics_output_{model_type}.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: metrics_output_{model_type}.png")
        
        # Calculate accuracy metrics
        accuracy_metrics = calculate_model_accuracy_metrics(model_type, config_path, checkpoint_path)
        if accuracy_metrics:
            all_metrics[model_name] = accuracy_metrics
            print(f"  PSNR: {accuracy_metrics['psnr_mean']:.2f} ± {accuracy_metrics['psnr_std']:.2f} dB")
            print(f"  SSIM: {accuracy_metrics['ssim_mean']:.4f} ± {accuracy_metrics['ssim_std']:.4f}")
            print(f"  LPIPS: {accuracy_metrics['lpips_mean']:.4f} ± {accuracy_metrics['lpips_std']:.4f}")
        else:
            print(f"  No checkpoint available for accuracy calculation")
    
    # Plot accuracy summary
    if all_metrics:
        fig3 = plot_accuracy_summary(all_metrics)
        if fig3:
            fig3.savefig('metrics_accuracy_summary.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved: metrics_accuracy_summary.png")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - metrics_training_*.png (Training curves)")
    print("  - metrics_output_*.png (Model outputs)")
    print("  - metrics_accuracy_summary.png (Accuracy comparison)")
    print("\nOpen these files to see your model performance!")

if __name__ == '__main__':
    main()

