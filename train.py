import torch
import torch.nn as nn
from DiffusionNet import DiffusionNet
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import logging
from saliency_metric import cal_fm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime  # 导入 datetime 模块

logging.basicConfig(filename='training.log', level=logging.INFO)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(30),  # 随机旋转
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

valid_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载 DIS5K 数据集
root_dir = '/data/caoxizhe/MVANet_cxz/data/DIS5K'  # DIS5K 数据集路径

# 创建数据集实例
train_dataset = DIS5KDataset(root_dir=root_dir, phase='DIS-TR', transform=train_transform)
valid_dataset = DIS5KDataset(root_dir=root_dir, phase='DIS-VD', transform=valid_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
num_valid_batches = len(valid_loader)
num_train_batches = len(train_loader)


# 定义模型
model = DiffusionNet().to(device)

# 使用并行训练
model = nn.DataParallel(model)

total_epochs = 80
warmup_epochs = 5  # 前5个epoch为warmup阶段
learning_rate = 2e-3
save_interval = 10 # 每10个epoch保存一次模型
valid_interval = 5  # 每5个epoch验证一次

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义学习率调度器
warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.1, 
    end_factor=1.0, 
    total_iters=warmup_epochs
)

main_scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=(total_epochs - warmup_epochs),
    eta_min=1e-7  # 最小学习率，可自行设置
)

scheduler = SequentialLR(
    optimizer, 
    schedulers=[warmup_scheduler, main_scheduler], 
    milestones=[warmup_epochs]
)

# 加载模型和优化器状态
def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def save_output(output, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # output = output * 0.5 + 0.5  # 反归一化

    # 遍历批次中的每一张图片并保存
    for j in range(output.size(0)):
        single_image = output[j].squeeze(0)  # 去除通道维度
        output_image = transforms.ToPILImage('L')(single_image.cpu())
        output_filepath = os.path.join(output_dir, f'epoch_{epoch}_picrure_{j}.png')
        output_image.save(output_filepath)

def train(run_id, start_epoch=0):

    # 获取当前日期并格式化为字符串
    current_date = datetime.now().strftime('%Y-%m-%d')

    logging.info(f"Training started on {current_date} with run_id {run_id}")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0

        for i, (image, gt) in enumerate(train_loader):

            image, gt = image.to(device), gt.to(device)
            optimizer.zero_grad()

            # 前向传播
            loss = model(image, gt, training=True)
            loss = loss.mean()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算并输出当前训练进度百分比
            progress = (i + 1) / len(train_loader) * 100
            print(f"Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{num_train_batches}], Progress: {progress:.2f}%")

            
            if i % 10 == 9:  # 每10个batch打印一次损失
                print(f'Epoch [{epoch + 1}/{total_epochs}], Batch [{i + 1}/{num_train_batches}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
            


        # 每save_interval个epoch保存一次模型
        if (epoch + 1) % save_interval == 0:
            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, f'checkpoints/{current_date}_run_{run_id}_checkpoint_epoch_{epoch + 1}.pth')

            print(f'Model saved at epoch {epoch + 1}')
        
        '''
        # 验证模型
        if (epoch + 1) % valid_interval == 0:
            print("validation_epoch_start")
            model.eval()
        
            with torch.no_grad():
                cal_fm_instance = cal_fm(num_valid_batches)

                for i, (image, gt) in enumerate(valid_loader):

                    # 计算并输出当前进度百分比
                    progress = (i + 1) / num_valid_batches * 100
                    print(f"Batch [{i+1}/{num_valid_batches}], Progress: {progress:.2f}%")

                    image, gt = image.to(device), gt.to(device)
                    output = model(image, gt=None, training=False)
                    cal_fm_instance.update(gt, output)

                    valid_output_dir = f'valid_output_images/{current_date}_run_{run_id}'  # 在文件夹名称中添加日期和训练次数

                    save_output(output, epoch, valid_output_dir)
            
                cal_fm_instance.compute_fmeasure()
                meanF, maxF = cal_fm_instance.get_results()

                logging.info(f'Epoch [{epoch + 1}/{total_epochs}], Mean F-measure: {meanF.mean():.4f}, Max F-measure: {maxF.mean():.4f}')
        '''
    
        # 更新学习率
        scheduler.step()


    print('Training finished.')

if __name__ == "__main__":
    # 如果有保存的模型，加载并继续训练
    start_epoch = 0
    checkpoint_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")
    run_id = 1
    train(run_id, start_epoch)