import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class ContentLoss(nn.Module):
    """L1 content loss."""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return self.criterion(pred_features, target_features)

class GradientPenaltyLoss(nn.Module):
    """Gradient penalty for WGAN-GP."""
    def __init__(self):
        super().__init__()
    
    def forward(self, discriminator, real, fake, device):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        disc_interpolates = discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

