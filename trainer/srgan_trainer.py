import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Dict
from .base_trainer import BaseTrainer
from utils.losses import ContentLoss, PerceptualLoss
from utils import metrics
from models.srgan import SRGAN

class SRGANTrainer(BaseTrainer):
    def __init__(self, config: Dict, device: torch.device):
        model = SRGAN(config)
        super().__init__(config, model, device)
        self.g_optimizer, self.d_optimizer = model.get_optimizers()
        self.content_loss = ContentLoss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_adv = config['loss_weights']['adversarial']
        self.lambda_content = config['loss_weights']['content']
        self.lambda_perceptual = config['loss_weights']['perceptual']
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
        self.model.generator.train(); self.model.discriminator.train()
        metrics = {'g_loss':0.0,'d_loss':0.0,'content_loss':0.0,'perceptual_loss':0.0,'adv_loss':0.0}
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(self.device); hr_imgs = hr_imgs.to(self.device)

            # Discriminator step
            self.d_optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                sr = self.model.generator(lr_imgs)
                real_pred = self.model.discriminator(hr_imgs)
                fake_pred = self.model.discriminator(sr.detach())
                d_loss = 0.5 * (torch.mean((real_pred - 1)**2) + torch.mean(fake_pred**2))
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.d_optimizer)
            self.scaler_d.update()

            # Generator step
            self.g_optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                sr = self.model.generator(lr_imgs)
                pred = self.model.discriminator(sr)
                adv_loss = 0.5 * torch.mean((pred - 1)**2)
                c_loss = self.content_loss(sr, hr_imgs)
                p_loss = self.perceptual_loss(sr, hr_imgs)
                g_loss = self.lambda_adv * adv_loss + self.lambda_content * c_loss + self.lambda_perceptual * p_loss
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.step(self.g_optimizer)
            self.scaler_g.update()

            metrics['g_loss'] += float(g_loss.detach().cpu())
            metrics['d_loss'] += float(d_loss.detach().cpu())
            metrics['content_loss'] += float(c_loss.detach().cpu())
            metrics['perceptual_loss'] += float(p_loss.detach().cpu())
            metrics['adv_loss'] += float(adv_loss.detach().cpu())

        for k in metrics: metrics[k] /= len(dataloader)
        return metrics

    def validate(self, dataloader):
        self.model.generator.eval()
        val = {'psnr':0.0,'ssim':0.0,'lpips':0.0}
        with torch.no_grad():
            for lr_imgs, hr_imgs in dataloader:
                lr_imgs = lr_imgs.to(self.device); hr_imgs = hr_imgs.to(self.device)
                sr = self.model.generator(lr_imgs)
                val['psnr'] += metrics.calculate_psnr(sr, hr_imgs, self.config)
                val['ssim'] += metrics.calculate_ssim(sr, hr_imgs, self.config)
                val['lpips'] += metrics.calculate_lpips(sr, hr_imgs)
        for k in val: val[k] /= len(dataloader)
        return val

    def generate_samples(self, num_samples: int):
        self.model.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['data']['num_channels'], self.config['data']['lr_size'], self.config['data']['lr_size']).to(self.device)
            return self.model.generator(z)
