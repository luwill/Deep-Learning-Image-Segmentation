# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 4 章 — (无标号片段)
# 原始位置：book-review/drafts/ch04-v3.md:435
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefineNetBlock, self).__init__()
        self.rcu1 = ResidualConvUnit(in_channels)
        self.rcu2 = ResidualConvUnit(out_channels)
        self.mrf = MultiResolutionFusion(out_channels, in_channels, out_channels)
        self.crp = ChainedResidualPooling(out_channels)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = self.mrf(*[self.rcu1(inp) for inp in inputs])
        x = self.crp(x)
        return self.output_conv(self.rcu2(x))


class RefineNet(nn.Module):
    def __init__(self, num_classes):
        super(RefineNet, self).__init__()
        resnet = resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5])
        self.layer3 = nn.Sequential(*list(resnet.children())[6])
        self.layer4 = nn.Sequential(*list(resnet.children())[7])

        self.refine3 = RefineNetBlock(1024, 256)
        self.refine2 = RefineNetBlock(512, 256)
        self.refine1 = RefineNetBlock(256, 256)
        self.refine0 = RefineNetBlock(64, 256)
        self.output_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.layer1(x)
        x1 = self.layer2(x0)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        r3 = self.refine3(x3)
        r2 = self.refine2(x2, r3)
        r1 = self.refine1(x1, r2)
        r0 = self.refine0(x0, r1)
        out = self.output_conv(r0)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
