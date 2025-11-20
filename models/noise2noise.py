import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_filters = config['model'].get('num_filters', 64)
        num_residual_blocks = config['model'].get('num_residual_blocks', 8)
        num_channels = config['data'].get('num_channels', 3)
        
        # Encoder
        self.conv_input = nn.Conv2d(num_channels, num_filters, 9, padding=4)
        
        # Residual blocks
        self.residual_layers = nn.Sequential(*[
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Decoder
        self.conv_mid = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_filters)
        self.conv_output = nn.Conv2d(num_filters, num_channels, 9, padding=4)
    
    def forward(self, x):
        out = F.relu(self.conv_input(x))
        residual = out
        out = self.residual_layers(out)
        out = self.bn_mid(self.conv_mid(out))
        out = out + residual
        out = torch.tanh(self.conv_output(out))
        return out

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_filters = config['model'].get('num_filters', 64)
        num_channels = config['data'].get('num_channels', 3)
        
        def conv_block(in_filters, out_filters, stride=1, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            conv_block(num_channels, num_filters, bn=False),
            conv_block(num_filters, num_filters * 2, stride=2),
            conv_block(num_filters * 2, num_filters * 4, stride=2),
            conv_block(num_filters * 4, num_filters * 8, stride=2),
            conv_block(num_filters * 8, num_filters * 8, stride=2),
            nn.Conv2d(num_filters * 8, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)

class Noise2NoiseGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.config = config
    
    def get_optimizers(self):
        lr = self.config['training'].get('learning_rate', 0.0001)
        beta1 = self.config['training'].get('beta1', 0.9)
        beta2 = self.config['training'].get('beta2', 0.999)
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        return g_optimizer, d_optimizer

