# Training Guide

This guide explains how to train all models in this project.

## Quick Start

### Train All Models
```bash
python scripts/train_all.py
```

This will train all three models (SRGAN, Noise2Noise, DeblurGAN) sequentially.

### Train Individual Models
```bash
# Train SRGAN
python scripts/train.py --config configs/srgan.yaml

# Train Noise2Noise
python scripts/train.py --config configs/noise2noise.yaml

# Train DeblurGAN
python scripts/train.py --config configs/deblurgan.yaml
```

## Data Setup

Place your training images in the following directories:
- `data/srgan/` - High-resolution images for SRGAN
- `data/noise2noise/` - Clean images for Noise2Noise
- `data/deblurgan/` - Sharp images for DeblurGAN

**Note:** If no images are found, the system will use synthetic data for training.

## Configuration

Edit the YAML files in `configs/` to adjust:
- Training parameters (epochs, batch size, learning rate)
- Model architecture (number of filters, residual blocks)
- Loss weights
- Data augmentation settings

## Checkpoints

Trained models are saved in:
- `checkpoints/srgan_model/`
- `checkpoints/noise2noise_model/`
- `checkpoints/deblurgan_model/`

Each checkpoint directory contains:
- `best.pth` - Best model based on validation metric
- `latest.pth` - Latest checkpoint
- `config.yaml` - Configuration used for training
- `metrics.json` - Training metrics history

## Monitoring

Training logs are saved in:
- `logs/srgan_model/`
- `logs/noise2noise_model/`
- `logs/deblurgan_model/`

## Requirements

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

For GPU training, ensure PyTorch with CUDA support is installed.

