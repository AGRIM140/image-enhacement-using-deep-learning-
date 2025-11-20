import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Dict
from .base_trainer import BaseTrainer
from utils.losses import ContentLoss, PerceptualLoss
from utils import metrics
from models.deblurgan import DeblurGAN

class DeblurGANTrainer(BaseTrainer):
    def __init__(self, config: Dict, device: torch.device):
        model = DeblurGAN(config)
        super().__init__(config, model, device)
        self.g_optimizer, self.d_optimizer = model.get_optimizers()
        self.content_loss = ContentLoss()
        self.perceptual_loss = PerceptualLoss()
        w = config['loss_weights']
        self.lambda_adv = w.get('adversarial', 1.0)
        self.lambda_content = w.get('content', 1.0)
        self.lambda_perceptual = w.get('perceptual', 1.0)
        self.lambda_gp = w.get('gradient_penalty', 10.0)
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
        stats = {'g_loss':0.0,'d_loss':0.0,'content_loss':0.0,'perceptual_loss':0.0,'adv_loss':0.0,'gp':0.0}
        for blurred, sharp in dataloader:
            blurred = blurred.to(self.device); sharp = sharp.to(self.device)

            # Discriminator step
            self.d_optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                fake_det = self.model.generator(blurred).detach()
                real_pred = self.model.discriminator(sharp)
                fake_pred = self.model.discriminator(fake_det)
                # LSGAN as default
                d_loss = 0.5 * (torch.mean((real_pred - 1)**2) + torch.mean(fake_pred**2))
                gp = metrics.calculate_gradient_penalty(self.model.discriminator, sharp, fake_det, self.device)
                d_loss = d_loss + self.lambda_gp * gp
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.d_optimizer)
            self.scaler_d.update()

            # Generator step
            self.g_optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                fake = self.model.generator(blurred)
                fake_pred = self.model.discriminator(fake)
                g_adv = 0.5 * torch.mean((fake_pred - 1)**2)
                c_loss = self.content_loss(fake, sharp)
                p_loss = self.perceptual_loss(fake, sharp)
                g_loss = self.lambda_adv * g_adv + self.lambda_content * c_loss + self.lambda_perceptual * p_loss
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.step(self.g_optimizer)
            self.scaler_g.update()

            stats['g_loss'] += float(g_loss.detach().cpu())
            stats['d_loss'] += float(d_loss.detach().cpu())
            stats['content_loss'] += float(c_loss.detach().cpu())
            stats['perceptual_loss'] += float(p_loss.detach().cpu())
            stats['adv_loss'] += float(g_adv.detach().cpu())
            stats['gp'] += float(gp.detach().cpu())

        for k in stats: stats[k] /= len(dataloader)
        return stats

    def validate(self, dataloader):
        self.model.eval()
        v = {'psnr':0.0,'ssim':0.0,'lpips':0.0}
        with torch.no_grad():
            for blurred, sharp in dataloader:
                blurred = blurred.to(self.device); sharp = sharp.to(self.device)
                out = self.model.generator(blurred)
                v['psnr'] += metrics.calculate_psnr(out, sharp, self.config)
                v['ssim'] += metrics.calculate_ssim(out, sharp, self.config)
                v['lpips'] += metrics.calculate_lpips(out, sharp)
        for k in v: v[k] /= len(dataloader)
        return v

    def generate_samples(self, num_samples: int):
        self.model.eval()
        with torch.no_grad():
            rnd = torch.randn(num_samples, self.config['data']['num_channels'],
                               self.config['data']['image_size'], self.config['data']['image_size']).to(self.device)
            return self.model.generator(rnd)
