# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 4 章 — (无标号片段)
# 原始位置：book-review/drafts/ch04-v3.md:334
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SegNetDec(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = vgg16.features
        self.enc1 = features[0:4]
        self.enc2 = features[5:9]
        self.enc3 = features[10:16]
        self.enc4 = features[17:23]
        self.enc5 = features[24:-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.dec5 = SegNetDec(512, 512, 1)
        self.dec4 = SegNetDec(512, 256, 1)
        self.dec3 = SegNetDec(256, 128, 1)
        self.dec2 = SegNetDec(128, 64, 0)
        self.final = nn.Sequential(
            nn.Conv2d(64, classes, 3, padding=1),
            nn.BatchNorm2d(classes), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        e1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.enc2(e1)
        e2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.enc3(e2)
        e3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.enc4(e3)
        e4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.enc5(e4)
        e5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        d5 = self.dec5(F.max_unpool2d(e5, m5, kernel_size=2, stride=2, output_size=x5.size()))
        d4 = self.dec4(F.max_unpool2d(d5, m4, kernel_size=2, stride=2, output_size=x4.size()))
        d3 = self.dec3(F.max_unpool2d(d4, m3, kernel_size=2, stride=2, output_size=x3.size()))
        d2 = self.dec2(F.max_unpool2d(d3, m2, kernel_size=2, stride=2, output_size=x2.size()))
        d1 = F.max_unpool2d(d2, m1, kernel_size=2, stride=2, output_size=x1.size())
        return self.final(d1)
