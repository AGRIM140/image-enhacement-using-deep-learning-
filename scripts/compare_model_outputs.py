#!/usr/bin/env python3
"""
Compare model outputs with ground truth and calculate accuracy metrics.
Shows before/after comparisons with quality scores.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from tools.model_runner import ModelRunner
from utils import metrics

def tensor_to_numpy(tensor):
    """Convert tensor to numpy for visualization."""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.permute(1, 2, 0).cpu().numpy()

def generate_test_pair(model_type, config):
    """Generate input/target pair for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        input_img = torch.nn.functional.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    
    return input_img, target

def evaluate_model_output(model_type, config_path, checkpoint_path, num_samples=5):
    """Evaluate model and create comparison visualization."""
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    config = yaml.safe_load(open(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    runner = ModelRunner(model_type, config, device)
    runner.load_checkpoint(checkpoint_path)
    runner.generator.eval()
    
    # Collect results
    results = []
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    for i in range(num_samples):
        input_img, target = generate_test_pair(model_type, config)
        
        with torch.no_grad():
            output = runner.generator(input_img)
        
        # Calculate metrics
        psnr = metrics.calculate_psnr(output, target, config)
        ssim = metrics.calculate_ssim(output, target, config)
        lpips = metrics.calculate_lpips(output, target)
        
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips)
        
        results.append({
            'input': input_img.cpu(),
            'output': output.cpu(),
            'target': target.cpu(),
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips
        })
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{model_type.upper()} - Model Output Comparison\n'
                 f'Average PSNR: {np.mean(all_psnr):.2f} dB | '
                 f'SSIM: {np.mean(all_ssim):.4f} | '
                 f'LPIPS: {np.mean(all_lpips):.4f}',
                 fontsize=14, fontweight='bold')
    
    for i, result in enumerate(results):
        input_np = tensor_to_numpy(result['input'])
        output_np = tensor_to_numpy(result['output'])
        target_np = tensor_to_numpy(result['target'])
        
        axes[i, 0].imshow(input_np)
        axes[i, 0].set_title(f'Sample {i+1}: Input', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(output_np)
        axes[i, 1].set_title(f'Model Output\nPSNR: {result["psnr"]:.2f} dB\n'
                           f'SSIM: {result["ssim"]:.4f}\n'
                           f'LPIPS: {result["lpips"]:.4f}', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target_np)
        axes[i, 2].set_title(f'Ground Truth', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig, {
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'lpips_mean': np.mean(all_lpips),
        'lpips_std': np.std(all_lpips)
    }

def main():
    """Main comparison function."""
    print("="*60)
    print("MODEL OUTPUT COMPARISON AND ACCURACY")
    print("="*60)
    print()
    
    models = [
        ('srgan', 'configs/srgan.yaml', 'checkpoints/srgan_model/best.pth'),
        ('noise2noise', 'configs/noise2noise.yaml', 'checkpoints/noise2noise_model/best.pth'),
        ('deblurgan', 'configs/deblurgan.yaml', 'checkpoints/deblurgan_model/best.pth'),
    ]
    
    all_results = {}
    
    for model_type, config_path, checkpoint_path in models:
        print(f"\nEvaluating {model_type.upper()}...")
        
        if not Path(checkpoint_path).exists():
            print(f"  Skipping - checkpoint not found: {checkpoint_path}")
            continue
        
        fig, metrics_dict = evaluate_model_output(model_type, config_path, checkpoint_path, num_samples=5)
        
        if fig:
            filename = f'accuracy_comparison_{model_type}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename}")
            print(f"  PSNR: {metrics_dict['psnr_mean']:.2f} ± {metrics_dict['psnr_std']:.2f} dB")
            print(f"  SSIM: {metrics_dict['ssim_mean']:.4f} ± {metrics_dict['ssim_std']:.4f}")
            print(f"  LPIPS: {metrics_dict['lpips_mean']:.4f} ± {metrics_dict['lpips_std']:.4f}")
            all_results[model_type] = metrics_dict
    
    if all_results:
        # Create summary comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(all_results.keys())
        psnr_means = [all_results[m]['psnr_mean'] for m in model_names]
        ssim_means = [all_results[m]['ssim_mean'] for m in model_names]
        lpips_means = [all_results[m]['lpips_mean'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.6
        
        axes[0].bar(x, psnr_means, width, color='green', alpha=0.7)
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('Peak Signal-to-Noise Ratio\n(Higher is Better)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(x, ssim_means, width, color='blue', alpha=0.7)
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('Structural Similarity Index\n(Higher is Better)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(x, lpips_means, width, color='red', alpha=0.7)
        axes[2].set_ylabel('LPIPS')
        axes[2].set_title('Learned Perceptual Image Patch Similarity\n(Lower is Better)')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig('accuracy_comparison_summary.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: accuracy_comparison_summary.png")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()

