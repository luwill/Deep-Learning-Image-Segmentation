# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 3 章 — 代码 3-4 残差块
# 原始位置：book-review/drafts/ch03-v3.md:358
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入 torch 相关模块
import torch
from torch import nn
import torch.nn.functional as F

# 定义残差块类
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=[1, 1], downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride[0],
            padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride[1],
            padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    # 定义残差块的前向计算流程
    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        # 残差连接
        out = out + residual
        out = F.relu(out)
        return out
