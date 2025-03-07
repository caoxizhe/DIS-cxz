import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from DiffusionNet import DiffusionNet
from train import DIS5KDataset
from datetime import datetime  # 导入 datetime 模块
import cv2
import numpy as np

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载测试集
root_dir = '/data/caoxizhe/MVANet_cxz/data/DIS5K'  # DIS5K 数据集路径
test_dataset = DIS5KDataset(root_dir=root_dir, phase='DIS-debug', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义模型
model = DiffusionNet().to(device)

# 加载模型权重
checkpoint_path = 'checkpoints/2025-03-07_run_1_checkpoint_epoch_10.pth'
checkpoint = torch.load(checkpoint_path)

# 当使用 DataParallel 或 DistributedDataParallel 时,模型的 state_dict 会在键名前加上 module. 前缀。
# 处理 state_dict，去掉 'module.' 前缀
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
    else:
        new_state_dict[k] = v

# 加载模型权重
model.load_state_dict(new_state_dict)

# 创建保存生成图片的目录
# 获取当前日期并格式化为字符串
test_id = 1  # 运行 ID
current_date = datetime.now().strftime('%Y-%m-%d')
output_dir = f'generated_images/{current_date}_test_{test_id}'  
os.makedirs(output_dir, exist_ok=True)

# 生成图片并保存
model.eval()
with torch.no_grad():
    for i, (image, gt) in enumerate(test_loader):   
        # 计算并输出当前进度百分比
        progress = (i + 1) / len(test_loader) * 100
        print(f'Generating images: {progress:.2f}% ({i + 1}/{len(test_loader)})', end='\r')

        image = image.to(device)
        gt = gt.to(device)

        output = model(image, gt, training=False)

        # output = output * 0.5 + 0.5

        # 对 output 进行阈值处理，将其值映射为 0 或 1
        threshold = output.mean()
        output = torch.where(output > threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

        output = output.squeeze(0)

        # 获取 image_path
        image_path = test_dataset.image_paths[i]

        # 确保文件名包含扩展名
        output_filename = os.path.basename(image_path)
        output_filepath = os.path.join(output_dir, output_filename)

        # 将生成的图片保存到新目录下
        output_image = transforms.ToPILImage()(output.squeeze().cpu())
        output_image.save(output_filepath)

print('Test finished.')