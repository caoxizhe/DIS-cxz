import torch
from DiffusionNet import DiffusionNet # 导入模型

# 定义张量的形状
batch_size = 1  # 批次大小
input_channels = 3  # 输入通道数，例如 RGB 图像
height = 1024  # 图像高度
width = 1024  # 图像宽度

# 随机生成四维张量
image1 = torch.randn(batch_size, input_channels, height, width)
gt = torch.randn(batch_size, 1, height, width)

print("train:")
model = DiffusionNet()  # 实例化模型
output = model(image1, gt, training=True)  # 获得输出
print("train finish")  # 打印输出

print("test:")
image2 = torch.randn(batch_size, input_channels, height, width)

output = model(image2, gt=None, training=False)  # 获得输出
print(output.shape)  # 打印输出形状
print("test finish") 