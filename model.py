import torch
import torch.nn as nn
from DiffusionNet import DiffusionNet
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# 使用Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice

criterion = DiceLoss()

class DIS5KDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        image_dir = os.path.join(root_dir, phase, 'im')
        mask_dir = os.path.join(root_dir, phase, 'gt')
        for img_name in os.listdir(image_dir):  # img为.jpg格式,但mask为.png格式
            mask_name = img_name.split('.')[0] + '.png'
            self.image_paths.append(os.path.join(image_dir, img_name))
            self.mask_paths.append(os.path.join(mask_dir, mask_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 定义数据增强变换
train_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(30),  # 随机旋转
    transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载 DIS5K 数据集
root_dir = '/data/caoxizhe/MVANet_cxz/data/DIS5K'  # DIS5K 数据集路径

# 创建数据集实例
train_dataset = DIS5KDataset(root_dir=root_dir, phase='DIS-TR', transform=train_transform)
valid_dataset = DIS5KDataset(root_dir=root_dir, phase='DIS-VD', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


# 定义模型
model = DiffusionNet()  

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 训练循环
num_epochs = 20
save_interval = 5  # 每5个epoch保存一次模型
valid_interval = 5  # 每5个epoch验证一次

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (image, gt) in enumerate(train_loader):
        optimizer.zero_grad()

        # 前向传播
        output = model(image, gt, training=True)
        loss = criterion(output, gt)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:  # 每10个batch打印一次损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # 每save_interval个epoch保存一次模型
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        print(f'Model saved at epoch {epoch + 1}')

    # 验证模型
    if (epoch + 1) % valid_interval == 0:
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for image, gt in valid_loader:
                output = model(image)
                loss = criterion(output, gt)
                valid_loss += loss.item()

        print(f'Validation Loss: {valid_loss / len(valid_loader):.4f}')
    
    # 更新学习率
    scheduler.step()


print('Training finished.')