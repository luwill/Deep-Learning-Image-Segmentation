# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 4 章 — (无标号片段)
# 原始位置：book-review/drafts/ch04-v3.md:671
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()         # ✅ 修正 super 类名（原稿误写为 UNetPlusPlus）
        # 编码器
        self.conv0_0 = ConvBlock(in_channels, 64)
        self.conv1_0 = ConvBlock(64, 128)
        self.conv2_0 = ConvBlock(128, 256)
        self.conv3_0 = ConvBlock(256, 512)
        self.conv4_0 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 嵌套卷积块
        self.conv0_1 = ConvBlock(64 + 128, 64)
        self.conv1_1 = ConvBlock(128 + 256, 128)
        self.conv2_1 = ConvBlock(256 + 512, 256)
        self.conv3_1 = ConvBlock(512 + 1024, 512)
        self.conv0_2 = ConvBlock(64 * 2 + 128, 64)
        self.conv1_2 = ConvBlock(128 * 2 + 256, 128)
        self.conv2_2 = ConvBlock(256 * 2 + 512, 256)
        self.conv0_3 = ConvBlock(64 * 3 + 128, 64)
        self.conv1_3 = ConvBlock(128 * 3 + 256, 128)
        self.conv0_4 = ConvBlock(64 * 4 + 128, 64)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return self.final_conv(x0_4)
