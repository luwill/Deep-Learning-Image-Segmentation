# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:932
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name='resnet101',
    encoder_weights='imagenet',
    in_channels=3,
    classes=21,                  # PASCAL VOC 类别数
)
