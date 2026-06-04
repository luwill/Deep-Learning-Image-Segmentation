# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 7 章 — 代码 7-1 3D 卷积最简使用示例
# 原始位置：book-review/drafts/ch07-v3.md:120
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn

# 输入张量：batch=1, channels=4, D×H×W=64×64×64
inputs = torch.randn(1, 4, 64, 64, 64)

# 创建 3D 卷积层：输入通道 4，输出通道 12，卷积核 3×3×3
conv3d = nn.Conv3d(in_channels=4, out_channels=12, kernel_size=3)

# 对输入做卷积
output = conv3d(inputs)
print(output.shape)
# 输出：torch.Size([1, 12, 62, 62, 62])
