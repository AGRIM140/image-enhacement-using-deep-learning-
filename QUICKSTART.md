# Quick Start Guide

## Training All Models

To train all models in the project:

```bash
python scripts/train_all.py
```

This will train:
1. **SRGAN** - Super-resolution GAN
2. **Noise2Noise** - Image denoising GAN
3. **DeblurGAN** - Image deblurring GAN

### Individual Model Training

```bash
# Train SRGAN
python scripts/train.py --config configs/srgan.yaml

# Train Noise2Noise
python scripts/train.py --config configs/noise2noise.yaml

# Train DeblurGAN
python scripts/train.py --config configs/deblurgan.yaml
```

## Data Setup

Place training images in:
- `data/srgan/` - High-resolution images
- `data/noise2noise/` - Clean images
- `data/deblurgan/` - Sharp images

**Note:** If directories are empty, synthetic data will be generated automatically.

## Deploy Web App

### Using Docker (Recommended)

**Windows (PowerShell):**
```powershell
.\deploy.ps1
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

Or manually:
```bash
docker-compose build
docker-compose up -d
```

The web app will be available at: **http://localhost:8501**

### Without Docker

```bash
streamlit run webui/streamlit_app.py
```

## Using the Web App

1. Open http://localhost:8501 in your browser
2. Select a model type (SRGAN, Noise2Noise, DeblurGAN)
3. Optionally specify a checkpoint path (e.g., `checkpoints/srgan_model/best.pth`)
4. Upload an image
5. Click "Enhance" to process the image

## Checkpoints

After training, checkpoints are saved in:
- `checkpoints/srgan_model/best.pth`
- `checkpoints/noise2noise_model/best.pth`
- `checkpoints/deblurgan_model/best.pth`

Use these paths in the web app to load trained models.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support, ensure PyTorch with CUDA is installed.

