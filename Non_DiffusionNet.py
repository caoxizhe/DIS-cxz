from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from diffusion_utilities import *
from diffusers import AutoencoderKL, DDIMScheduler
import cv2

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义边缘损失函数
def edge_loss(pred, target):
    """
    计算边缘损失
    :param pred: 预测图像 (batch_size, channels, height, width)
    :param target: 目标图像 (batch_size, channels, height, width)
    :return: 边缘损失
    """
    pred_edges = []
    target_edges = []
    
    for i in range(pred.shape[0]):
        pred_img = pred[i].squeeze().detach().cpu().numpy()
        target_img = target[i].squeeze().detach().cpu().numpy()
        
        pred_edge = cv2.Canny((pred_img * 255).astype(np.uint8), 100, 200)
        target_edge = cv2.Canny((target_img * 255).astype(np.uint8), 100, 200)
        
        pred_edges.append(torch.tensor(pred_edge, device=pred.device).float() / 255.0)
        target_edges.append(torch.tensor(target_edge, device=target.device).float() / 255.0)
    
    pred_edges = torch.stack(pred_edges).unsqueeze(1)
    target_edges = torch.stack(target_edges).unsqueeze(1)
    
    return F.mse_loss(pred_edges, target_edges)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim)       # Query来自掩膜特征
        self.to_kv = nn.Linear(context_dim, 2*query_dim)  # Key/Value来自图像特征
        self.proj = nn.Linear(query_dim, query_dim)       # 输出投影

    def forward(self, x, context):
        # x: [B, C, H, W] (掩膜特征)
        # context: [B, C', H', W'] (图像特征)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]

        # 提取图像特征的Key和Value
        context_flat = context.view(B, -1, context.shape[1])  # [B, H'*W', C']
        k, v = self.to_kv(context_flat).chunk(2, dim=-1)      # [B, H'*W', C], [B, H'*W', C]

        # 计算注意力权重
        q = self.to_q(x_flat)  # [B, H*W, C]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H*W, H'*W']
        attn = attn.softmax(dim=-1)

        # 加权融合Value
        out = attn @ v  # [B, H*W, C]
        out = self.proj(out).permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        return out + x  # 残差连接


# UNet 模型,此部分参照DeepLearningAI: how diffusion model works课程代码的框架
class UNet(nn.Module):
    def __init__(self, in_channels=4, n_feat=256, image_size=64):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.image_size = image_size

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.CrossAttention1 = CrossAttention(n_feat, 4)
        self.CrossAttention2 = CrossAttention(2 * n_feat, 4)
        self.CrossAttention3 = CrossAttention(2 * n_feat, 2 * n_feat)
        self.CrossAttention4 = CrossAttention(n_feat, n_feat)
  

        self.cond_down1 = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.cond_down2 = UnetDown(n_feat, n_feat)
        self.cond_down3 = UnetDown(n_feat, 2 * n_feat)

        # Zero conv vs nn.Conv2d
        self.cond_down_zeroconv_1_0 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_1_1 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_2_0 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_2_1 = nn.Conv2d(n_feat , n_feat , 1 , 1 ,0)
        self.cond_down_zeroconv_3_0 = nn.Conv2d(2 * n_feat , 2 * n_feat , 1 , 1, 0)
        self.cond_down_zeroconv_3_1 = nn.Conv2d(2 * n_feat , 2 * n_feat , 1 , 1, 0)  

        self.to_vec = nn.Sequential(nn.AvgPool2d((2)), nn.GELU())

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
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, image):
        """
        image : (batch, 4, 64, 64) : input image
        t : (batch, 1) : time step
        """
    
        # pass the input image through the initial convolutional layer
        x = self.init_conv(image)   # [1, n_feat, 64, 64]
        # pass the result through the down-sampling path

        down1 = self.down1(x)       #[1, n_feat, 32, 32]
        down1 = self.CrossAttention1(down1, image)  # [1, n_feat, 32, 32]
        down2 = self.down2(down1)   #[1, n_feat * 2, 16, 16]
        down2 = self.CrossAttention2(down2, image)  # [1, n_feat * 2, 16, 16]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)  # [1, n_feat * 2, 8, 8]

        cond_down1 = self.cond_down1(image)  # [1, n_feat, 64, 64]
        cond_down2 = self.cond_down2(cond_down1)    # [1, n_feat * 2, 32, 32]
        cond_down3 = self.cond_down3(cond_down2)    # [1, n_feat * 2, 16, 16]


        up1 = self.up0(hiddenvec)   # [1, n_feat * 2, 16, 16]
        up1 = self.CrossAttention3(up1, self.cond_down_zeroconv_3_0(cond_down3)) + self.cond_down_zeroconv_3_1(cond_down3)   # [1, n_feat * 2, 16, 16]

        up2 = self.up1(up1, down2)  # [1, n_feat, 32, 32]
        up2 = self.CrossAttention4(up2, self.cond_down_zeroconv_2_0(cond_down2)) + self.cond_down_zeroconv_2_1(cond_down2)   # [1, n_feat, 32, 32]

        up3 = self.up2(up2, down1) * self.cond_down_zeroconv_1_0(cond_down1) + self.cond_down_zeroconv_1_1(cond_down1)    # [1, 4, 64, 64]

        out = self.out(torch.cat((up3, x), 1))  # [1, 4, 64, 64]
        return out

class Non_DiffusionNet(nn.Module):
    def __init__(self):  
        super(Non_DiffusionNet, self).__init__()

        self.vae = AutoencoderKL.from_pretrained(
            "/data/caoxizhe/pretrained_vae", 
        )
        '''
        accepts (b, c, 512, 512)
        returns (b, 4, 64, 64)
        '''
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.UNet = UNet()


    def loss(self, pred_mask, mask):
        """
        pred_noise : (batch, 4, 64, 64) : predicted noise
        noise : (batch, 4, 64, 64) : ground truth noise
        """
        # 计算损失
        loss = F.mse_loss(pred_mask, mask) + 20 * edge_loss(pred_mask, mask)

        return loss

    

    def forward(self, image, gt, training):
        '''
        具体pipeline如下:
        1. Encode the input image & ground truth
        2. add noise to the latent ground truth
        3. Concatenate the latent image and latent ground truth(channel-wise)
        4. pass the input image through the UNet, together with time step and latent image embedding(similar to context embedding in diffusion model)
        5. denoise the image
        6. Decode the output
        '''

        if training == True:
        
            with torch.no_grad():

                gt = gt.repeat(1, 3, 1, 1)  # shape [b, 3, 512, 512]
                
                latent_image = self.vae.encode(image).latent_dist.mean.to(device)  # shape [b, 4, 64, 64]

                latent_gt = self.vae.encode(gt).latent_dist.mean.to(device)  # shape [b, 4, 64, 64]

            pred_gt = self.UNet(latent_image).to(device) # shape [b, 4, 64, 64]

            loss = self.loss(pred_gt, latent_gt)

            return loss

        else:
            
            latent_image = self.vae.encode(image).latent_dist.mean.to(device)  # shape [b, 4, 64, 64]

            pred_gt = self.UNet(latent_image).to(device) # shape [b, 4, 64, 64]

            batched_mask_3_ch = self.vae.decode(pred_gt).sample
            # 3 to 1
            batched_mask = torch.mean(batched_mask_3_ch, dim=1, keepdim=True)   # shape [b, 1, 512, 512]
            return batched_mask
            