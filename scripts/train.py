import argparse
import yaml
import torch
from pathlib import Path
from trainer import srgan_trainer, noise2noise_trainer, deblurgan_trainer
from data.datasets import get_dataloader

def load_config(p):
    return yaml.safe_load(open(p))

def get_trainer(cfg, device):
    mt = cfg['model']['type']
    if mt == 'srgan':
        return srgan_trainer.SRGANTrainer(cfg, device)
    if mt == 'noise2noise':
        return noise2noise_trainer.Noise2NoiseTrainer(cfg, device)
    if mt == 'deblurgan':
        return deblurgan_trainer.DeblurGANTrainer(cfg, device)
    raise ValueError(f"Unsupported model type: {mt}")

def main():
    parser = argparse.ArgumentParser(description='Train GAN models')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = get_trainer(cfg, device)
    model_type = cfg['model']['type']
    print(f"Training {model_type} model...")
    
    # Create data loaders
    train_loader = get_dataloader(cfg, model_type, is_train=True)
    val_loader = get_dataloader(cfg, model_type, is_train=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train
    num_epochs = args.epochs or cfg['training'].get('num_epochs', cfg['training'].get('epochs', 100))
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
