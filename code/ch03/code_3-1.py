# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 3 章 — 代码 3-1 SegNet VGG16 编码器
# 原始位置：book-review/drafts/ch03-v3.md:102
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入 torchvision 的预训练模型模块
from torchvision import models
import torch.nn as nn

# 定义 SegNet 类（仅包含编码器部分）
class SegNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        # 获取 VGG16 训练权重模型结构
        vgg16 = models.vgg16(pretrained=True)
        # 获取 VGG16 网络层特征
        features = vgg16.features
        # 1-4 层作为第一个编码块
        self.enc1 = features[0:4]
        # 5-8 层作为第二个编码块
        self.enc2 = features[5:9]
        # 9-15 层作为第三个编码块
        self.enc3 = features[10:16]
        # 16-22 层作为第四个编码块
        self.enc4 = features[17:23]
        # 23 到最后一层作为第五个编码块
        self.enc5 = features[24:]  # ⚠️ EDIT: 原为 features[24:-1]，会跳过最后的 MaxPool2d，与 SegNet 设计不一致
