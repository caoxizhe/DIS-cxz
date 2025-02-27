from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from diffusion_utilities import *
from diffusers import AutoencoderKL

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
    x_0, t = x_0.to("cuda"), t.to("cuda")
    
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)   
    
    # 生成随机噪声
    noise = torch.randn_like(x_0)
    
    # 加噪公式:
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t , noise

# 定义去除噪声的函数
def denoise(x_t, t, pred_noise):
    """
    去除输入图像 x_t 的噪声
    :param x_t: 加噪后的图像 (batch_size, channels, height, width)
    :param t: 时间步
    :param pred_noise: 预测的噪声 (batch_size, channels, height, width)
    :return: 去噪后的图像
    """

    x_t, t, pred_noise = x_t.to("cuda"), t.to("cuda"), pred_noise.to("cuda")

    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    mean = (x_t - pred_noise * sqrt_one_minus_alphas_cumprod_t) / sqrt_alphas_cumprod_t
    return mean


# UNet 模型,此部分参照DeepLearningAI: how diffusion model works课程代码的框架
class UNet(nn.Module):
    def __init__(self, in_channels=8, n_feat=256, image_size=64):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.image_size = image_size

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)       
        self.down2 = UnetDown(n_feat, 2 * n_feat)    


        self.cond_down1 = ResidualConvBlock(in_channels // 2, n_feat, is_res=True)
        self.cond_down2 = UnetDown(n_feat, n_feat)
        self.cond_down3 = UnetDown(n_feat, 2 * n_feat)

        # Zero conv vs nn.Conv2d
        self.cond_down_zeroconv_1_0 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_1_1 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_2_0 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_2_1 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_3_0 = nn.Conv2d(2 * n_feat , 2 * n_feat , 1 , 1, 0)
        self.cond_down_zeroconv_3_1 = nn.Conv2d(2 * n_feat , 2 * n_feat , 1 , 1, 0)

        # Convert the feature maps to a vector and apply an activation

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.image_size//16, self.image_size//16), # up-sample  
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
            #nn.Conv2d(n_feat, self.in_channels//2, 3, 1, 1), # map to same number of channels as input
            nn.Conv2d(n_feat, self.in_channels//2 , 1 ,1, 0)
        )

    def forward(self, image, t, latent_image):
        """
        image : (batch, 4, 64, 64) : input image
        t : (batch, 1) : time step
        latent_image_flat : (batch, 16*32*32)    : image embedding
        """
    
        # pass the input image through the initial convolutional layer
        x = self.init_conv(image)   
        # pass the result through the down-sampling path

        down1 = self.down1(x)       
        down2 = self.down2(down1)  
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)  
        # embed context and timestep

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1) 

        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        cond_down1 = self.cond_down1(latent_image)
        cond_down2 = self.cond_down2(cond_down1)
        cond_down3 = self.cond_down3(cond_down2)

        up1 = self.up0(hiddenvec)*self.cond_down_zeroconv_3_0(cond_down3) + self.cond_down_zeroconv_3_1(cond_down3)
        up2 = self.up1(up1+temb1, down2) * self.cond_down_zeroconv_2_0(cond_down2) + self.cond_down_zeroconv_2_1(cond_down2)
        up3 = self.up2(up2+temb2, down1) * self.cond_down_zeroconv_1_0(cond_down1) + self.cond_down_zeroconv_1_1(cond_down1)    
        out = self.out(torch.cat((up3, x), 1))  
        return out

class DiffusionNet(nn.Module):
    def __init__(self):  
        super(DiffusionNet, self).__init__()

        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="vae",
        )
        self.vae.eval()
        self.vae.requires_grad_(False)
        #embed [B,4,64,64] to [B,16*32*32]
        self.UNet = UNet()

    def loss(self, out, latent_gt, t):
        """
        out : (batch, 4, 64, 64) : denoised image
        latent_gt : (batch, 4, 64, 64) : ground truth image
        t : (batch, 1) : time step
        """
        # 计算损失
        loss = F.mse_loss(out, latent_gt)
        return loss

    def forward(self, image, gt, training):
        if training == True:
            gt = gt.repeat(1, 3, 1, 1)
        else :
            gt = gt.repeat(1, 3, 1, 1)
            #gt = torch.zeros_like(image)
        '''
        具体pipeline如下:
        1. Encode the input image & ground truth
        2. add noise to the latent ground truth
        3. Concatenate the latent image and latent ground truth(channel-wise)
        4. pass the input image through the UNet, together with time step and latent image embedding(similar to context embedding in diffusion model)
        5. denoise the image
        6. Decode the output
        '''
        # Encode the input image & ground truth

        with torch.no_grad():
            latent_image = self.vae.encode(image).latent_dist.mean  # shape [B, 4, 64, 64]
            latent_gt = self.vae.encode(gt).latent_dist.mean  # shape [B, 4, 64, 64]
            t = torch.ones(latent_gt.shape[0], 1,dtype=int).to(device) * 999  
            latent_gt_noisy , noise = add_noise(latent_gt, t) 
            latent_gt_noisy = latent_gt_noisy.to(device)
            noise = noise.to(device)
            x = torch.cat([latent_image, latent_gt_noisy], dim=1).to(device)   

        # add noise to the latent ground truth
        #t = torch.randint(0, T, (latent_gt.shape[0],1)).to(device) 
        # set t to 999
        #x = torch.cat([latent_gt_noisy, latent_gt_noisy], dim=1).to(device)   
        #print(latent_image.shape)
        #print(x.shape)
        #print(latent_image_flat.shape)
        # pass the input image through the UNet
        pred_noise = self.UNet(x, t/T, latent_image)
        out = denoise(latent_gt_noisy, t, pred_noise).to(device)

        if training == True:
            loss = self.loss(pred_noise, noise, t)
            #loss = self.loss(out, latent_gt, t)
            return loss
        else:
            batched_mask_3_ch = self.vae.decode(out).sample  
            # 3 to 1
            batched_mask = torch.mean(batched_mask_3_ch, dim=1, keepdim=True)
            return batched_mask

if __name__ == "__main__":
    model = DiffusionNet().to(device)
    input = torch.randn(1, 3, 512, 512).to(device)
    gt = torch.randn(1, 1, 512, 512).to(device)
    print(model(input, gt, training=True))
    print(model(input, gt, training=False).shape)