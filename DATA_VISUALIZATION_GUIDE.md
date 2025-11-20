# Data Visualization Guide

## Quick Visualization

### View All Data Samples
```bash
python scripts/visualize_data.py
```

This creates three PNG files:
- `data_visualization_srgan.png` - Shows LR/HR pairs
- `data_visualization_noise2noise.png` - Shows noisy/clean pairs
- `data_visualization_deblurgan.png` - Shows blurred/sharp pairs

### Interactive Viewing
```bash
python scripts/view_data_interactive.py
```

Opens matplotlib windows showing data samples interactively.

## What You'll See

### SRGAN Data
- **Left**: Low-resolution images (64x64)
- **Right**: High-resolution images (256x256)
- Shows the super-resolution task

### Noise2Noise Data
- **Left**: Noisy images (with synthetic noise)
- **Right**: Clean target images
- Shows the denoising task

### DeblurGAN Data
- **Left**: Blurred images
- **Right**: Sharp target images
- Shows the deblurring task

## Current Data Status

Since your data directories are empty, the visualizations show **synthetic data**:
- Randomly generated images
- Automatically created input/target pairs
- Used for training when no real data is available

## Understanding the Visualizations

1. **Input Images** (left column): What the model receives
2. **Target Images** (right column): What the model should produce
3. **Training Goal**: Learn to transform input → target

## After Adding Real Data

When you add real images to the data directories:
1. Run visualization again
2. You'll see your actual images instead of synthetic data
3. The visualizations will reflect your real dataset

## Tips

- **More samples**: Edit the script to change `num_samples` parameter
- **Save images**: The script automatically saves PNG files
- **Compare**: Run before/after adding real data to see the difference

## Files Generated

- `data_visualization_srgan.png`
- `data_visualization_noise2noise.png`
- `data_visualization_deblurgan.png`

These files show exactly what your models are training on!

