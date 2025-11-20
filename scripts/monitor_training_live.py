#!/usr/bin/env python3
"""
Live monitoring of training metrics from log files.
Updates automatically as training progresses.
"""
import matplotlib.pyplot as plt
import time
from pathlib import Path
import re
from collections import defaultdict

def parse_latest_log(log_file):
    """Parse the latest log file for metrics."""
    if not Path(log_file).exists():
        return None
    
    metrics = defaultdict(list)
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line in lines:
            # Extract epoch
            epoch_match = re.search(r'Epoch (\d+)', line)
            if not epoch_match:
                continue
            
            epoch = int(epoch_match.group(1))
            
            # Extract metrics with various patterns
            patterns = {
                'g_loss': r'g_loss[:\s]+([\d.]+)',
                'd_loss': r'd_loss[:\s]+([\d.]+)',
                'psnr': r'psnr[:\s]+([\d.]+)',
                'ssim': r'ssim[:\s]+([\d.]+)',
                'lpips': r'lpips[:\s]+([\d.]+)',
                'content_loss': r'content_loss[:\s]+([\d.]+)',
                'perceptual_loss': r'perceptual_loss[:\s]+([\d.]+)',
                'adversarial_loss': r'adversarial_loss[:\s]+([\d.]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    metrics[key].append((epoch, float(match.group(1))))
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None
    
    return dict(metrics) if metrics else None

def plot_live_metrics(model_name, log_file, update_interval=5):
    """Plot and update metrics in real-time."""
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Live Training Metrics', fontsize=16, fontweight='bold')
    
    while True:
        metrics = parse_latest_log(log_file)
        
        if not metrics:
            for ax in axes.flat:
                ax.clear()
                ax.text(0.5, 0.5, 'Waiting for training data...', 
                       ha='center', va='center', transform=ax.transAxes)
            plt.draw()
            plt.pause(update_interval)
            continue
        
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # Plot Generator Loss
        if 'g_loss' in metrics:
            epochs, losses = zip(*metrics['g_loss'])
            axes[0, 0].plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=3)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Generator Loss')
            axes[0, 0].grid(True, alpha=0.3)
            if losses:
                axes[0, 0].text(0.02, 0.98, f'Latest: {losses[-1]:.4f}', 
                             transform=axes[0, 0].transAxes, va='top', 
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot Discriminator Loss
        if 'd_loss' in metrics:
            epochs, losses = zip(*metrics['d_loss'])
            axes[0, 1].plot(epochs, losses, 'r-', linewidth=2, marker='o', markersize=3)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Discriminator Loss')
            axes[0, 1].grid(True, alpha=0.3)
            if losses:
                axes[0, 1].text(0.02, 0.98, f'Latest: {losses[-1]:.4f}', 
                             transform=axes[0, 1].transAxes, va='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot PSNR
        if 'psnr' in metrics:
            epochs, values = zip(*metrics['psnr'])
            axes[1, 0].plot(epochs, values, 'g-', linewidth=2, marker='o', markersize=3)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('PSNR (dB)')
            axes[1, 0].set_title('Validation PSNR (Higher is Better)')
            axes[1, 0].grid(True, alpha=0.3)
            if values:
                best = max(values)
                axes[1, 0].axhline(y=best, color='g', linestyle='--', alpha=0.5)
                axes[1, 0].text(0.02, 0.98, f'Best: {best:.2f} dB\nLatest: {values[-1]:.2f} dB', 
                             transform=axes[1, 0].transAxes, va='top',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Plot SSIM
        if 'ssim' in metrics:
            epochs, values = zip(*metrics['ssim'])
            axes[1, 1].plot(epochs, values, 'm-', linewidth=2, marker='o', markersize=3)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('SSIM')
            axes[1, 1].set_title('Validation SSIM (Higher is Better)')
            axes[1, 1].grid(True, alpha=0.3)
            if values:
                best = max(values)
                axes[1, 1].axhline(y=best, color='m', linestyle='--', alpha=0.5)
                axes[1, 1].text(0.02, 0.98, f'Best: {best:.4f}\nLatest: {values[-1]:.4f}', 
                             transform=axes[1, 1].transAxes, va='top',
                             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(update_interval)
        
        # Check if window is closed
        if not plt.get_fignums():
            break

def main():
    """Main monitoring function."""
    import sys
    
    models = [
        ('SRGAN', 'logs/srgan_model'),
        ('Noise2Noise', 'logs/noise2noise_model'),
        ('DeblurGAN', 'logs/deblurgan_model'),
    ]
    
    print("="*60)
    print("LIVE TRAINING MONITOR")
    print("="*60)
    print("\nAvailable models:")
    for i, (name, _) in enumerate(models, 1):
        print(f"  {i}. {name}")
    
    if len(sys.argv) > 1:
        try:
            choice = int(sys.argv[1])
            model_name, log_dir = models[choice - 1]
        except:
            print("Invalid choice, using first model")
            model_name, log_dir = models[0]
    else:
        model_name, log_dir = models[0]
        print(f"\nMonitoring {model_name} (use: python {sys.argv[0]} <1-3> for other models)")
    
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob('*.log'))
    
    if not log_files:
        print(f"\nNo log files found in {log_dir}")
        print("Waiting for training to start...")
        return
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"\nMonitoring: {model_name}")
    print(f"Log file: {latest_log}")
    print(f"\nPress Ctrl+C to stop monitoring")
    print("="*60)
    
    try:
        plot_live_metrics(model_name, latest_log, update_interval=5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    main()

