# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 4 章 — (无标号片段)
# 原始位置：book-review/drafts/ch04-v3.md:90
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
from torch import nn

# 指定输入向量
x = torch.rand(25088,)

# 定义全连接层
fc = nn.Linear(25088, 4096)

# 定义卷积层
conv2d = nn.Conv2d(512, 4096, 7)

# 全连接层计算
fc_output = fc(x)
print(fc_output.size())              # torch.Size([4096])

# 输入向量变换为 (B, C, H, W)
x = torch.reshape(x, (1, 512, 7, 7))

# 卷积层计算
conv_output = conv2d(x)
print(conv_output.size())            # torch.Size([1, 4096, 1, 1])
