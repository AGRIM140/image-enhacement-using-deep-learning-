import torch
from utils.image_utils import load_image, save_image
from pathlib import Path

def _get_model(model_type: str, config: dict):
    if model_type == 'srgan':
        from models.srgan import SRGAN
        return SRGAN(config)
    if model_type == 'deblurgan':
        from models.deblurgan import DeblurGAN
        return DeblurGAN(config)
    if model_type == 'noise2noise':
        from models.noise2noise import Noise2NoiseGAN
        return Noise2NoiseGAN(config)
    if model_type == 'cyclegan':
        from models.cyclegan import CycleGAN
        return CycleGAN(config)
    raise ValueError(f"Unsupported model_type: {model_type}")

class ModelRunner:
    def __init__(self, model_type: str, config: dict, device: torch.device):
        self.device = device
        self.config = config
        self.model_type = model_type
        self.model = _get_model(model_type, config)
        # prefer attribute 'generator' for inference if present
        if hasattr(self.model, 'generator'):
            self.generator = self.model.generator
        elif hasattr(self.model, 'generator_A2B'):
            self.generator = self.model.generator_A2B
        else:
            self.generator = self.model
        self.generator.to(device)

    def load_checkpoint(self, path: str):
        ck = torch.load(path, map_location=self.device)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            # Some checkpoints save the entire wrapper model state - try to load into generator safely
            try:
                self.generator.load_state_dict(ck['model_state_dict'], strict=False)
            except Exception:
                # fallback: load state dict directly if it matches
                self.generator.load_state_dict(ck, strict=False)
        else:
            self.generator.load_state_dict(ck, strict=False)

    def enhance_and_save(self, input_path: str, output_path: str):
        img = load_image(input_path, config=self.config).to(self.device)
        self.generator.eval()
        with torch.no_grad():
            out = self.generator(img)
        save_image(out, output_path, config=self.config)
