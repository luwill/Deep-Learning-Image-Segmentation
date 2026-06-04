# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 5 章 — (无标号片段)
# 原始位置：book-review/drafts/ch05-v3.md:192
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = ASPPConv(in_channels, out_channels, dilation=6)
        self.conv3 = ASPPConv(in_channels, out_channels, dilation=12)
        self.conv4 = ASPPConv(in_channels, out_channels, dilation=18)
        self.pool = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        return x
