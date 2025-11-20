#!/usr/bin/env python3
"""
Script to check training status and progress.
"""
from pathlib import Path
import json
import os

def check_training_status():
    """Check the status of model training."""
    models = ['srgan_model', 'noise2noise_model', 'deblurgan_model']
    base_checkpoint_dir = Path('checkpoints')
    base_log_dir = Path('logs')
    
    print("=" * 60)
    print("TRAINING STATUS CHECK")
    print("=" * 60)
    print()
    
    for model_name in models:
        print(f"[{model_name.upper()}]")
        print("-" * 60)
        
        checkpoint_dir = base_checkpoint_dir / model_name
        log_dir = base_log_dir / model_name
        
        # Check checkpoints
        if checkpoint_dir.exists():
            latest = checkpoint_dir / 'latest.pth'
            best = checkpoint_dir / 'best.pth'
            metrics_file = checkpoint_dir / 'metrics.json'
            
            if latest.exists():
                print(f"  [OK] Latest checkpoint: {latest.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print(f"  [IN PROGRESS] No checkpoint yet (training in progress...)")
            
            if best.exists():
                print(f"  [OK] Best checkpoint: {best.stat().st_size / 1024 / 1024:.2f} MB")
            
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    print(f"  [OK] Metrics file found ({len(metrics)} entries)")
                    # Show latest metrics
                    if metrics:
                        latest_key = max(metrics.keys(), key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)
                        print(f"  [INFO] Latest metrics: {latest_key}")
                except:
                    print(f"  [WARNING] Metrics file exists but couldn't be read")
        else:
            print(f"  [IN PROGRESS] Checkpoint directory not created yet")
        
        # Check logs
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"  [OK] Latest log: {latest_log.name}")
                # Show last few lines
                try:
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"  [LOG] Last entry: {lines[-1].strip()[:80]}")
                except:
                    pass
            else:
                print(f"  [IN PROGRESS] No log files yet")
        else:
            print(f"  [IN PROGRESS] Log directory not created yet")
        
        print()
    
    print("=" * 60)
    print("TIP: Run this script periodically to monitor training progress")
    print("=" * 60)

if __name__ == '__main__':
    check_training_status()

