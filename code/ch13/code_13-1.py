# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — 代码 13-1 采用的是**离线增强**：在训练前一次性把每张原图扩充成多张并写盘，训练时直接读盘即可。这种做法在小数据集（如原始 ICH 仅 271 张）上能快速建立训练流程，但有两个缺点：① 磁盘空间膨胀 5 倍；② 每个 epoch 看到的"增强样本"是固定的有限张，**等价于扩充 5 倍数据集而非真随机增强**，可能对这固定数量的"伪样本"过拟合。
# 原始位置：book-review/drafts/ch13-v4.md:207
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.PadIfNeeded(min_height=512, min_width=512, p=1.0),
    A.RandomCrop(height=512, width=512, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})
