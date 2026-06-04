# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 4 章 — 代码 4-2 给出 FCN-8s 的一个 PyTorch 简略实现方式，便于读者加深对 FCN 的理解。代码中对于卷积下采样使用了 VGG-16 的预训练权重，分别构建了四个特征提取模块、一个卷积块和三个独立的卷积层。在前向传播流程中，将 conv7、pool3 和 pool4 进行融合，最后再做 8 倍的双线性插值上采样。
# 原始位置：book-review/drafts/ch04-v3.md:135
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FCN8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 提取 VGG-16 预训练权重作为特征
        feats = list(models.vgg16(pretrained=True).features.children())
        # 取前 9 层为第一特征模块
        self.feat1 = nn.Sequential(*feats[0:9])
        # 第二特征模块
        self.feat2 = nn.Sequential(*feats[10:16])
        # 第三特征模块
        self.feat3 = nn.Sequential(*feats[17:23])
        # 第四特征模块
        self.feat4 = nn.Sequential(*feats[24:30])

        # 卷积层权重不参与训练更新
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout(),
        )

        # 把最后三层全连接改为 1×1 卷积
        self.conv1 = nn.Conv2d(256, num_classes, 1)
        self.conv2 = nn.Conv2d(512, num_classes, 1)
        self.conv3 = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feat1 = self.feat1(x)
        feat2 = self.feat2(feat1)
        feat3 = self.feat3(feat2)
        feat4 = self.feat4(feat3)
        conv_blocks = self.conv_blocks(feat4)

        conv1 = self.conv1(feat2)
        conv2 = self.conv2(feat3)
        conv3 = self.conv3(conv_blocks)

        outputs = F.upsample_bilinear(conv_blocks, conv2.size()[2:])
        # 第一次融合
        outputs = outputs + conv2
        outputs = F.upsample_bilinear(outputs, conv1.size()[2:])
        # 第二次融合
        outputs = outputs + conv1
        return F.upsample_bilinear(outputs, x.size()[2:])
