
# GAN Image Enhancement Project

This repository contains implementations, trainers, utilities and a Streamlit UI for multiple GAN-based image enhancement models:
- SRGAN (super-resolution)
- DeblurGAN (deblurring)
- Noise2Noise (denoising)
- CycleGAN (color/contrast enhancement)

## What's included
- `models/` : model implementations
- `trainer/` : trainer skeletons for each model
- `utils/` : image, metrics, and loss utilities
- `webui/streamlit_app.py` : Streamlit app for inference
- `tools/model_runner.py` : small inference helper used by the UI
- `configs/` : your uploaded YAML config files
- `checkpoints/` : (directory to store checkpoints)
- Deployment files: `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `run.sh`, `start_train.sh`

## Quickstart (local)

1. Create and activate virtualenv (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run Streamlit UI
```bash
streamlit run webui/streamlit_app.py
```
Open http://localhost:8501

4. Train (example)
```bash
python scripts/train.py --config configs/srgan.yaml
```
(Adjust `--config` to whichever model config you want.)

5. Evaluate
```bash
python scripts/evaluate.py --config configs/srgan.yaml --checkpoint checkpoints/srgan/latest.pth --num_samples 16
```

## Docker (recommended for deployment)
Build and run:
```bash
docker build -t gan-enhancer:latest .
docker run -p 8501:8501 gan-enhancer:latest
```
Or use docker-compose:
```bash
docker-compose up --build
```

## Notes
- Place checkpoints for models under `checkpoints/<model_name>/`.
- Config files in `configs/` control dataset, training and model hyperparameters.
- If your GPU is available in the container, pass `--gpus` to `docker run` (Docker 19+).
