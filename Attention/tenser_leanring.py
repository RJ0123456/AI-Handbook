# tenser operations learning of pytorch
import torch
# 创建两个张量
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
# 张量加法
c = a + b
print("Addition:\n", c)
# 张量乘法
d = a * b
print("Element-wise Multiplication:\n", d)
# 矩阵乘法
e = a @ b
print("Matrix Multiplication:\n", e)
# 张量转置
f = a.t()
print("Transpose of a:\n", f)
# 张量转置（使用transpose方法）
f1 = a.transpose(0, 1)
print("original a:\n", a)
print("Transpose of a (using transpose method):\n", f1)
# 张量求和
g = a.sum()
print("Sum of a:\n", g)
# 张量的形状
print("Shape of a:\n", a.shape)
# 张量的维度
print("Number of dimensions of a:\n", a.ndim)
# 张量的大小
print("Size of a:\n", a.size())
# 张量的类型
print("Data type of a:\n", a.dtype)
# 张量的设备
print("Device of a:\n", a.device)

# for more tensor operations, please refer to the official PyTorch documentation: https://pytorch.org/docs/stable/torch.html
# three dimensional tensor
x = torch.rand(2, 3, 4)  # (batch_size, seq_len, d_model)
print("Shape of x:", x.shape)  # (2, 3, 4)
print("Number of dimensions of x:", x.ndim)  # 3
print("Size of x:", x.size())  # (2, 3, 4)
print("Data type of x:", x.dtype)  # torch.float32
print("Device of x:", x.device)  # cpu
