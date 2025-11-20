#!/usr/bin/env python3
"""
Script to train all models in the project sequentially.
"""
import yaml
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import load_config, get_trainer
from data.datasets import get_dataloader

MODELS = [
    ('configs/srgan.yaml', 'SRGAN'),
    ('configs/noise2noise.yaml', 'Noise2Noise'),
    ('configs/deblurgan.yaml', 'DeblurGAN'),
]

def train_model(config_path, model_name):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = get_trainer(cfg, device)
    model_type = cfg['model']['type']
    
    # Create data loaders
    train_loader = get_dataloader(cfg, model_type, is_train=True)
    val_loader = get_dataloader(cfg, model_type, is_train=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train
    num_epochs = cfg['training'].get('num_epochs', cfg['training'].get('epochs', 100))
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    print(f"\n{model_name} training completed!\n")

def main():
    print("Starting training for all models...")
    print(f"Total models to train: {len(MODELS)}\n")
    
    for config_path, model_name in MODELS:
        try:
            train_model(config_path, model_name)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            print(f"Skipping {model_name} and continuing...\n")
            continue
    
    print("\n" + "="*60)
    print("All models training completed!")
    print("="*60)

if __name__ == '__main__':
    main()

