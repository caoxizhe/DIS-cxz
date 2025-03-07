import torch
from DiffusionNet import DiffusionNet # 导入模型
from Non_DiffusionNet import Non_DiffusionNet # 导入模型

# 查看 GPU 数量
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")

# 查看每台 GPU 的名称
for i in range(gpu_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义张量的形状
batch_size = 1  # 批次大小
input_channels = 3  # 输入通道数，例如 RGB 图像
height = 512  # 图像高度
width = 512  # 图像宽度

# 随机生成四维张量
image1 = torch.randn(batch_size, input_channels, height, width).to(device)
gt = torch.randn(batch_size, 1, height, width).to(device)

print("train:")
# model = DiffusionNet().to(device)  # 实例化模型
model = Non_DiffusionNet().to(device)  # 实例化模型
output = model(image1, gt, training=True)  # 获得输出
print("train finish")  # 打印输出

print("test:")
image2 = torch.randn(batch_size, input_channels, height, width).to(device)

output = model(image2, gt=None, training=False)  # 获得输出
print(output.shape)  # 打印输出形状
print("test finish") 