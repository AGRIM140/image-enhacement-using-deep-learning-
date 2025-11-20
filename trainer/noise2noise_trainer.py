import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Dict
from .base_trainer import BaseTrainer
from utils.losses import ContentLoss, PerceptualLoss
from utils import metrics
from models.noise2noise import Noise2NoiseGAN
import torch.nn as nn

class Noise2NoiseTrainer(BaseTrainer):
    def __init__(self, config: Dict, device: torch.device):
        model = Noise2NoiseGAN(config)
        super().__init__(config, model, device)
        self.g_optimizer, self.d_optimizer = model.get_optimizers()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.content_criterion = ContentLoss()
        self.perceptual_criterion = PerceptualLoss()
        self.l1_criterion = nn.L1Loss()
        self.lambda_adv = config['loss_weights']['adversarial']
        self.lambda_content = config['loss_weights']['content']
        self.lambda_perceptual = config['loss_weights']['perceptual']
        self.lambda_l1 = config['loss_weights']['l1']
        self.use_amp = config.get('use_amp', True)
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)
    
    def get_additional_checkpoint_state(self):
        return {
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'scaler_g': self.scaler_g.state_dict(),
            'scaler_d': self.scaler_d.state_dict()
        }
    
    def load_additional_checkpoint_state(self, state: dict):
        if 'g_optimizer' in state:
            self.g_optimizer.load_state_dict(state['g_optimizer'])
        if 'd_optimizer' in state:
            self.d_optimizer.load_state_dict(state['d_optimizer'])
        if 'scaler_g' in state:
            self.scaler_g.load_state_dict(state['scaler_g'])
        if 'scaler_d' in state:
            self.scaler_d.load_state_dict(state['scaler_d'])

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_metrics = {'g_loss':0.0,'d_loss':0.0,'content_loss':0.0,'perceptual_loss':0.0,'adversarial_loss':0.0,'l1_loss':0.0}
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs = noisy_imgs.to(self.device); clean_imgs = clean_imgs.to(self.device)

            # Discriminator
            self.d_optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                denoised = self.model.generator(noisy_imgs)
                real_preds = self.model.discriminator(clean_imgs)
                d_loss_real = self.adversarial_criterion(real_preds, torch.ones_like(real_preds))
                fake_preds = self.model.discriminator(denoised.detach())
                d_loss_fake = self.adversarial_criterion(fake_preds, torch.zeros_like(fake_preds))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.d_optimizer)
            self.scaler_d.update()

            # Generator
            self.g_optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                denoised = self.model.generator(noisy_imgs)
                fake_preds = self.model.discriminator(denoised)
                g_loss_adv = self.adversarial_criterion(fake_preds, torch.ones_like(fake_preds))
                content_loss = self.content_criterion(denoised, clean_imgs)
                perceptual_loss = self.perceptual_criterion(denoised, clean_imgs)
                l1_loss = self.l1_criterion(denoised, clean_imgs)
                g_loss = (self.lambda_adv * g_loss_adv + self.lambda_content * content_loss +
                          self.lambda_perceptual * perceptual_loss + self.lambda_l1 * l1_loss)
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.step(self.g_optimizer)
            self.scaler_g.update()

            epoch_metrics['g_loss'] += float(g_loss.detach().cpu())
            epoch_metrics['d_loss'] += float(d_loss.detach().cpu())
            epoch_metrics['content_loss'] += float(content_loss.detach().cpu())
            epoch_metrics['perceptual_loss'] += float(perceptual_loss.detach().cpu())
            epoch_metrics['adversarial_loss'] += float(g_loss_adv.detach().cpu())
            epoch_metrics['l1_loss'] += float(l1_loss.detach().cpu())

        for k in epoch_metrics: epoch_metrics[k] /= len(dataloader)
        return epoch_metrics

    def validate(self, dataloader):
        self.model.eval()
        val_metrics = {'psnr':0.0,'ssim':0.0,'lpips':0.0,'l1_loss':0.0}
        with torch.no_grad():
            for noisy_imgs, clean_imgs in dataloader:
                noisy_imgs = noisy_imgs.to(self.device); clean_imgs = clean_imgs.to(self.device)
                denoised = self.model.generator(noisy_imgs)
                val_metrics['psnr'] += metrics.calculate_psnr(denoised, clean_imgs, self.config)
                val_metrics['ssim'] += metrics.calculate_ssim(denoised, clean_imgs, self.config)
                val_metrics['lpips'] += metrics.calculate_lpips(denoised, clean_imgs)
                val_metrics['l1_loss'] += nn.L1Loss()(denoised, clean_imgs).item()
        for k in val_metrics: val_metrics[k] /= len(dataloader)
        return val_metrics

    def generate_samples(self, num_samples: int):
        self.model.eval()
        with torch.no_grad():
            noisy = torch.randn(num_samples, self.config['data']['num_channels'],
                                 self.config['data']['image_size'], self.config['data']['image_size']).to(self.device)
            return self.model.generator(noisy)
