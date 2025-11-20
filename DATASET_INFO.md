# Dataset Information

## Current Status

**All data directories are currently empty.** The training system will automatically use synthetic data.

## Synthetic Data (Current Setup)

When directories are empty, the system generates:
- **SRGAN**: Random high-resolution and low-resolution image pairs
- **Noise2Noise**: Random clean images with synthetic noise
- **DeblurGAN**: Random sharp images with synthetic blur

**Dataset size**: 1000 samples per model (synthetic)

## Adding Real Datasets (Optional)

If you want to use real images for better model quality:

### SRGAN Dataset
**Location**: `data/srgan/`

**Requirements**:
- High-resolution images (JPG, PNG)
- Any size (will be cropped/resized during training)
- Recommended: 100+ images for good results

**Example sources**:
- DIV2K dataset
- Flickr2K dataset
- Your own high-quality images

### Noise2Noise Dataset
**Location**: `data/noise2noise/`

**Requirements**:
- Clean, high-quality images (JPG, PNG)
- The system will automatically add noise during training
- Recommended: 100+ images

**Example sources**:
- ImageNet samples
- Your own clean images
- Any high-quality image dataset

### DeblurGAN Dataset
**Location**: `data/deblurgan/`

**Requirements**:
- Sharp, clear images (JPG, PNG)
- The system will automatically apply blur during training
- Recommended: 100+ images

**Example sources**:
- GoPro dataset (for motion deblurring)
- Your own sharp images
- Any high-quality image dataset

## How to Add Images

1. Download or collect your images
2. Place them directly in the appropriate `data/{model}/` directory
3. Supported formats: `.jpg`, `.jpeg`, `.png` (case-insensitive)
4. No subdirectories needed - just place images directly in the folder
5. Run training - the system will automatically detect and use them

## Training with Real vs Synthetic Data

**Synthetic Data (Current)**:
- ✅ Works immediately
- ✅ Good for testing the training pipeline
- ❌ Models may not generalize well to real images
- ❌ Limited diversity

**Real Data (Recommended for production)**:
- ✅ Better model quality
- ✅ Better generalization
- ✅ More realistic results
- ❌ Requires dataset collection/preparation

## Quick Start

You can start training right now with synthetic data:

```bash
python scripts/train_all.py
```

The system will automatically handle the empty directories and generate synthetic data.

