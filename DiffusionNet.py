from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from diffusion_utilities import *

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
T = 1000  # 扩散步数
beta_start = 1e-4  # 噪声调度起始值
beta_end = 0.02  # 噪声调度结束值

# 定义噪声调度(线性调度)
betas = (beta_end - beta_start) * torch.linspace(0, 1, T + 1) + beta_start
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)  # 累积乘积

# 定义添加噪声的函数
def add_noise(x_0, t):
    """
    为输入图像 x_0 添加噪声
    :param x_0: 原始图像 (batch_size, channels, height, width)
    :param t: 时间步 
    :return: 加噪后的图像和噪声
    """

    x_0, t = x_0.to("cuda:0"), t.to("cuda:0")
    
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)   
    
    # 生成随机噪声
    noise = torch.randn_like(x_0)
    
    # 加噪公式:
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t

def denoise(x_t, t, pred_noise):

    x_t, pred_noise = x_t.to("cuda:0"), pred_noise.to("cuda:0")

    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    mean = (x_t - pred_noise * sqrt_one_minus_alphas_cumprod_t) / sqrt_alphas_cumprod_t
    return mean

class VAEEncoder(nn.Module):
    def __init__(self, input_channels=3):
        """
        VAE编码器
        :param input_channels: 输入图像的通道数(默认3,表示RGB图像)
        """
        super(VAEEncoder, self).__init__()
        
        # 卷积层逐步下采样
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),  # 16x512x512
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),             # 32x256x256
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # 64x128x128
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入图像，形状为 (batch_size, channels, height, width)
        :return: 潜空间内图像
        """
        # 通过卷积层提取特征
        x = self.conv_layers(x)  # 输出形状: (batch_size, 64, 128, 128)
        
        return x

class VAEDecoder(nn.Module):
    def __init__(self, output_channels=1):
        """
        VAE解码器
        :param output_channels: 输出图像的通道数(默认3,表示RGB图像)
        """
        super(VAEDecoder, self).__init__()
        
        # 逐步上采样反卷积层
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x256x256
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16x512x512
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),  # 3x1024x1024
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入图像，形状为 (batch_size, channels, height, width)
        :return: 重构的图像
        """
        # 通过反卷积层生成重构图像
        x = self.deconv_layers(x)  # 输出形状: (batch_size, 3, 1024, 1024)
        
        return x

# UNet 模型
class UNet(nn.Module):
    def __init__(self, in_channels=128, n_feat=256, image_size=128):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.image_size = image_size

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[batch_size, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[batch_size, 256, 4, 4]

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(16*32*32, 2*n_feat)
        self.contextembed2 = EmbedFC(16*32*32, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.image_size//32, self.image_size//32), # up-sample  
            nn.GroupNorm(8, 2 * n_feat), # normalize                       
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), 
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels//2, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, image, t, latent_image_flat):
        """
        image : (batch, 128, 128, 128) : input image
        t : (batch, 1) : time step
        latent_image_flat : (batch, 16*32*32)    : image embedding
        """
    
        # pass the input image through the initial convolutional layer
        x = self.init_conv(image)   # [1, 256, 128, 128]
        # pass the result through the down-sampling path

        down1 = self.down1(x)       #[1, 256, 64, 64]
        down2 = self.down2(down1)   #[1, 256, 32, 32]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)  # [1, 512, 8, 8]

        # embed context and timestep
        cemb1 = self.contextembed1(latent_image_flat).view(-1, self.n_feat * 2, 1, 1)   # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)  # (batch, 2*n_feat, 1,1)
        cemb2 = self.contextembed2(latent_image_flat).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)


        up1 = self.up0(hiddenvec)  # [1, 512, 32, 32]
        up2 = self.up1(cemb1*up1 + temb1, down2)  # [1, 256, 64, 64]
        up3 = self.up2(cemb2*up2 + temb2, down1)    # [1, 256, 128, 128]
        out = self.out(torch.cat((up3, x), 1))  # [1, 128, 128, 128]
        return out

class DiffusionNet(nn.Module):
    def __init__(self):  
        super(DiffusionNet, self).__init__()

        self.VAEEncoder = VAEEncoder()
        self.VAEEncoder_gt = VAEEncoder(input_channels=1)
        self.VAEDecoder = VAEDecoder()
        self.UNet = UNet()

        self.conv1 = nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1)

    def forward(self, image, gt, training):

        if training == True:
            # Encode the input image & ground truth
            latent_image = self.VAEEncoder(image)   # (batch, 64, 128, 128)
            latent_gt = self.VAEEncoder_gt(gt)        # (batch, 64, 128, 128)

            # add noise to the latent ground truth
            t = torch.randint(0, T, (latent_gt.shape[0],1)).to(device)   # sample a random time step
             
            latent_gt_noisy = add_noise(latent_gt, t).to(device)  # (batch, 64, 128, 128)

            # Concatenate the latent image and latent ground truth(channel-wise)
            x = torch.cat([latent_image, latent_gt_noisy], dim=1).to(device)   # (batch, 128, 128, 128)

            latent_image = self.conv1(latent_image)  # (batch, 32, 64, 64)
            latent_image = self.conv2(latent_image)  # (batch, 16, 32, 32)

            # 展平空间维度
            latent_image_flat = latent_image.flatten(start_dim=1).to(device)  #  (batch, 16*32*32)

            # pass the input image through the UNet
            out = self.UNet(x, t/T, latent_image_flat)

            return out

        else:
            # Encode the input image
            latent_image = self.VAEEncoder(image)   # (batch, 64, 128, 128)

            # make random noise with shape of latent_image
            random_noise = torch.randn_like(latent_image)

            # Concatenate the latent image and latent ground truth(channel-wise)
            x = torch.cat([latent_image, random_noise], dim=1).to(device)   # (batch, 128, 128, 128)

            latent_image = self.conv1(latent_image)  # (batch, 32, 64, 64)
            latent_image = self.conv2(latent_image)  # (batch, 16, 32, 32)

            # 展平空间维度
            latent_image_flat = latent_image.flatten(start_dim=1).to(device)  #  (batch, 16*32*32)
            
            t = torch.full((latent_image.shape[0], 1), T).to(device)   
        
            pred_noise = self.UNet(x, t/T, latent_image_flat)

            # denoise the image
            out = denoise(random_noise, T, pred_noise).to(device)

            # Decode the output
            output = self.VAEDecoder(out)

            return output   