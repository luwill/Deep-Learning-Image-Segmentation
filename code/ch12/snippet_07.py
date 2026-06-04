# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:242
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabHeadV3Plus(nn.Module):
    # <!-- EDIT-V4: CODE-M5 ASPP dilate 应跟随 output_stride 自适应 -->
    # DeepLab v3+ 原论文 §4 表 5：OS=16 用 (6,12,18)；OS=8 用 (12,24,36)。
    def __init__(self, in_channels, low_level_channels, num_classes,
                 output_stride=16, aspp_dilate=None):
        super(DeepLabHeadV3Plus, self).__init__()
        if aspp_dilate is None:
            if output_stride == 16:
                aspp_dilate = (6, 12, 18)
            elif output_stride == 8:
                aspp_dilate = (12, 24, 36)
            else:
                raise ValueError(f'Unsupported output_stride={output_stride}')

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ASPP 空洞空间金字塔池化
        self.aspp = ASPP(in_channels, aspp_dilate)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        return self.classifier(
            torch.cat([low_level_feature, output_feature], dim=1)
        )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
