# Metrics and Accuracy Visualization Guide

## Overview

This guide explains how to visualize training metrics and model accuracy.

## Available Scripts

### 1. Visualize Training Metrics
```bash
python scripts/visualize_metrics.py
```

**What it does:**
- Reads metrics from training logs and checkpoint files
- Creates training curves (losses, PSNR, SSIM)
- Shows model output samples
- Calculates accuracy metrics (PSNR, SSIM, LPIPS)

**Output files:**
- `metrics_training_*.png` - Training curves for each model
- `metrics_output_*.png` - Model input/output samples
- `metrics_accuracy_summary.png` - Accuracy comparison across models

### 2. Live Training Monitor
```bash
python scripts/monitor_training_live.py
```

**What it does:**
- Monitors training in real-time
- Updates plots automatically as training progresses
- Shows latest metrics values
- Updates every 5 seconds

**Usage:**
```bash
# Monitor SRGAN (default)
python scripts/monitor_training_live.py

# Monitor Noise2Noise
python scripts/monitor_training_live.py 2

# Monitor DeblurGAN
python scripts/monitor_training_live.py 3
```

### 3. Compare Model Outputs
```bash
python scripts/compare_model_outputs.py
```

**What it does:**
- Compares model outputs with ground truth
- Calculates accuracy metrics on test samples
- Shows side-by-side comparisons
- Creates accuracy summary charts

**Output files:**
- `accuracy_comparison_*.png` - Detailed comparisons for each model
- `accuracy_comparison_summary.png` - Summary across all models

## Understanding the Metrics

### Training Metrics

1. **Generator Loss**: How well the generator is performing
   - Lower is better
   - Should decrease over time

2. **Discriminator Loss**: How well the discriminator is performing
   - Should stabilize (not too high, not too low)
   - Indicates training balance

3. **PSNR (Peak Signal-to-Noise Ratio)**: Image quality metric
   - Higher is better
   - Typical range: 20-40 dB
   - >30 dB is considered good

4. **SSIM (Structural Similarity Index)**: Perceptual quality metric
   - Higher is better
   - Range: 0-1
   - >0.9 is considered excellent

5. **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual distance
   - Lower is better
   - Range: 0-1
   - <0.1 is considered very similar

### Accuracy Metrics

These metrics measure how close model outputs are to ground truth:

- **PSNR**: Measures pixel-level accuracy
- **SSIM**: Measures structural similarity
- **LPIPS**: Measures perceptual similarity (human perception)

## When to Use Each Script

### During Training
- Use `monitor_training_live.py` to watch progress in real-time
- Check logs periodically with `visualize_metrics.py`

### After Training
- Use `visualize_metrics.py` for complete analysis
- Use `compare_model_outputs.py` for detailed accuracy evaluation

### Comparing Models
- Run `compare_model_outputs.py` to see which model performs best
- Check `accuracy_comparison_summary.png` for quick comparison

## Interpreting Results

### Good Training Signs
- ✅ Generator loss decreasing
- ✅ PSNR increasing over epochs
- ✅ SSIM increasing over epochs
- ✅ Discriminator loss stabilizing

### Warning Signs
- ⚠️ Generator loss not decreasing
- ⚠️ Discriminator loss too high/low (training imbalance)
- ⚠️ Metrics not improving after many epochs

### Model Quality Indicators
- **Excellent**: PSNR > 30 dB, SSIM > 0.9, LPIPS < 0.1
- **Good**: PSNR 25-30 dB, SSIM 0.8-0.9, LPIPS 0.1-0.2
- **Fair**: PSNR 20-25 dB, SSIM 0.7-0.8, LPIPS 0.2-0.3
- **Poor**: PSNR < 20 dB, SSIM < 0.7, LPIPS > 0.3

## Tips

1. **Monitor regularly**: Check metrics every few epochs
2. **Compare checkpoints**: Test different checkpoints to find best model
3. **Use multiple metrics**: Don't rely on just one metric
4. **Visual inspection**: Always look at actual output images, not just numbers

## Files Generated

After running visualization scripts, you'll have:

- Training curves showing loss and validation metrics over time
- Model output samples showing input/output pairs
- Accuracy comparisons showing quantitative results
- Summary charts comparing all models

These visualizations help you:
- Understand training progress
- Identify best checkpoints
- Compare model performance
- Debug training issues

