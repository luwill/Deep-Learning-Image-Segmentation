# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 5 章 — 代码 5-1 给出了 PSPNet 网络结构中 PPM 的实现过程，PSPNet 完整实现可参考本书配套代码库。
# 原始位置：book-review/drafts/ch05-v3.md:94
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """
    金字塔池化模块（PPM）
    in_channels: 输入特征通道数
    pool_sizes: 金字塔池化窗口大小列表，例如 [1, 2, 3, 6]
    out_channels: 每个池化分支的输出通道数
    """
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for pool_size in pool_sizes
        ])

    def forward(self, x):
        input_size = x.size()[2:]
        ppm_outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=input_size,
                                      mode='bilinear', align_corners=False)
            ppm_outs.append(upsampled)
        return torch.cat(ppm_outs, dim=1)
