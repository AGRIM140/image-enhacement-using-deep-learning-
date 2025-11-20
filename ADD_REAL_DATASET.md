# Guide: Adding Real Datasets for Better Training

## Why Use Real Datasets?

While synthetic data works for testing, **real datasets** will significantly improve:
- Model quality and generalization
- Real-world performance
- Output image quality

## Step-by-Step Guide

### 1. Prepare Your Images

Collect or download images for each model type:

#### For SRGAN (Super-Resolution)
- **What you need**: High-resolution, high-quality images
- **Recommended**: 100-1000+ images
- **Sources**:
  - [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images)
  - [Flickr2K Dataset](https://github.com/LimBee/NTIRE2017)
  - Your own high-quality photos

#### For Noise2Noise (Denoising)
- **What you need**: Clean, high-quality images
- **Recommended**: 100-1000+ images
- **Sources**:
  - [ImageNet](https://www.image-net.org/) samples
  - [COCO Dataset](https://cocodataset.org/)
  - Your own clean photos

#### For DeblurGAN (Deblurring)
- **What you need**: Sharp, clear images
- **Recommended**: 100-1000+ images
- **Sources**:
  - [GoPro Dataset](https://github.com/SeungjunNah/DeepDeblur_release) (for motion blur)
  - Your own sharp photos
  - Any high-quality image collection

### 2. Download and Organize

#### Option A: Manual Download
1. Download images from the sources above
2. Extract/copy images to the appropriate folders:
   ```
   data/srgan/          ← Place HR images here
   data/noise2noise/    ← Place clean images here
   data/deblurgan/      ← Place sharp images here
   ```

#### Option B: Using Python Script (Recommended)
I'll create a helper script to download and organize datasets automatically.

### 3. Verify Images Are Detected

After adding images, verify they're detected:

```python
from pathlib import Path

# Check SRGAN
srgan_path = Path("data/srgan")
srgan_images = list(srgan_path.glob("*.jpg")) + list(srgan_path.glob("*.png"))
print(f"SRGAN images: {len(srgan_images)}")

# Check Noise2Noise
n2n_path = Path("data/noise2noise")
n2n_images = list(n2n_path.glob("*.jpg")) + list(n2n_path.glob("*.png"))
print(f"Noise2Noise images: {len(n2n_images)}")

# Check DeblurGAN
deblur_path = Path("data/deblurgan")
deblur_images = list(deblur_path.glob("*.jpg")) + list(deblur_path.glob("*.png"))
print(f"DeblurGAN images: {len(deblur_images)}")
```

### 4. Retrain with Real Data

Once you've added real images:

```bash
# Train all models with real data
python scripts/train_all.py

# Or train individually
python scripts/train.py --config configs/srgan.yaml
python scripts/train.py --config configs/noise2noise.yaml
python scripts/train.py --config configs/deblurgan.yaml
```

The system will automatically detect and use your real images instead of synthetic data.

## Quick Dataset Download Script

I can create a Python script to help you download popular datasets. Would you like me to create:

1. **Dataset downloader script** - Downloads DIV2K, GoPro, etc.
2. **Image validation script** - Checks image quality and format
3. **Dataset splitter** - Splits data into train/val sets

## Tips

1. **Image Format**: JPG, JPEG, PNG are supported (case-insensitive)
2. **Image Size**: Any size works - the system will resize/crop automatically
3. **Quality**: Higher quality images = better model performance
4. **Quantity**: More images = better generalization (but 100+ is a good start)
5. **Diversity**: Include varied scenes, objects, lighting conditions

## Current Training Status

Your models are currently training with **synthetic data**. Once training completes (or if you stop it), you can:

1. Add real datasets to the `data/` folders
2. Retrain the models
3. Compare results between synthetic and real data training

## Need Help?

When you're ready to add real datasets, just ask and I'll:
- Help you download specific datasets
- Create helper scripts for dataset management
- Guide you through the process step-by-step

