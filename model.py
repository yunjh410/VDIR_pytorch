import torch
from torch import nn
from utils import *
from torchvision import transforms

class Encoder(nn.Module):
    def __init__(self, channels=64):
        super(Encoder, self).__init__()
        self.conv1 = conv2d(3, channels)
        self.conv2 = conv2d(channels, channels)
        self.conv3 = conv2d(channels, channels)
        self.conv4 = conv2d(channels, channels)
        self.conv5 = conv2d(channels, channels)
        
        self.conv6 = conv2d(channels, 4)
        self.conv7 = conv2d(channels, 4)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        # Block 1
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.relu(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        
        mu = self.conv6(x)
        log_sig = self.conv7(x)
        return mu, log_sig


class Decoder(nn.Module):
    def __init__(self, channels=64):
        super(Decoder, self).__init__()
        self.conv1 = conv2d(4, channels)
        self.conv2 = conv2d(channels, channels)
        self.conv3 = conv2d(channels, channels)
        self.conv4 = conv2d(channels, channels)
        self.conv5 = conv2d(channels, 3)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()
    
    def forward(self, input):
        # Block 1
        x = self.conv1(input)
        x = self.upsample(x)
        x = self.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.relu(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.relu(x)
        
        # output
        output = self.conv5(x)
        return output
        
        
class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(channels, channels)
        self.conv2 = conv2d(channels, channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        out = self.conv2(x) + input
        return out


class RIRBlock(nn.Module):
    def __init__(self, channels=64):
        super(RIRBlock, self).__init__()
        self.conv1 = conv2d(channels, channels)
        self.ResBlock1 = ResBlock()
        self.ResBlock2 = ResBlock()
        self.ResBlock3 = ResBlock()
        self.ResBlock4 = ResBlock()
        self.ResBlock5 = ResBlock()
        
    def forward(self, input):
        # RIR consists of N ResBlocks
        x = self.ResBlock1(input)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.ResBlock5(x)
        out = self.conv1(x) + input
        return out


class Denoiser(nn.Module):
    def __init__(self, channels=64):
        super(Denoiser, self).__init__()
        self.conv0 = conv2d(7, channels)
        self.conv1 = conv2d(channels, channels)
        self.conv2 = conv2d(channels, 3)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.RIRBlock1 = RIRBlock()
        self.RIRBlock2 = RIRBlock()
        self.RIRBlock3 = RIRBlock()
        self.RIRBlock4 = RIRBlock()
        self.RIRBlock5 = RIRBlock()        
    
    def forward(self, input, latent_c):
        
        # concat with latent variable
        resized_c = self.upsample(latent_c)
        resized_c = nn.Upsample(size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)(resized_c)
        x = torch.cat([input, resized_c], dim=1)
        
        x1 = self.conv0(x)
        
        # Denoiser is consists of D RIR Blocks
        x = self.RIRBlock1(x1)
        x = self.RIRBlock2(x)
        x = self.RIRBlock3(x)
        x = self.RIRBlock4(x)
        x = self.RIRBlock5(x)
        
        x = self.conv1(x) + x1 # long skip connection
        out = self.conv2(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super(Discriminator, self).__init__()
                
        self.model = nn.Sequential(
            
            SpectralNorm(conv2d(3, channels)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels, channels, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels, channels*2)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*2, channels*2, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*2, channels*4)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*4, channels*4, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*4, channels*8)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*8, channels*8, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*8, channels*8)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*8, channels*8, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*8, 1))
        )
        
    def forward(self, input):      
        return self.model(input)     


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv2d(4, 64)
        self.conv2 = conv2d(64, 3)
        # self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.resize = torch.Tensor.resize_()
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x
        

class VDIR(nn.Module):
    def __init__(self, channels=64):
        super(VDIR, self).__init__()
        self._Encoder = Encoder()
        self._Decoder = Decoder()
        self._Denoiser = Denoiser()
        self._Estimator = Estimator()
        
        self.normal = torch.distributions.normal.Normal(0., 1.)
        
    def forward(self, input):
                
        # re-parametrization trick
        mu, log_sigma_sq = self._Encoder(input)
        eps = self.normal.sample(log_sigma_sq.shape)
        eps = eps.cuda()
        latent_c = eps*torch.exp(log_sigma_sq / 2) + mu
        
        # Decoder
        y_hat = self._Decoder(latent_c)
        
        # Denoiser
        res = self._Denoiser(input, latent_c)
        denoised = input - res
        
        # sigma Estimator
        sigma_est = self._Estimator(latent_c)
        
        return mu, log_sigma_sq, y_hat, denoised, sigma_est