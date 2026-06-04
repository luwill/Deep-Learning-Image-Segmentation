# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:73
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 计算训练集 mean / std，结果用于 12.3 节的 transforms.Normalize
import numpy as np
from PIL import Image
from pathlib import Path

mean = np.zeros(3)
std = np.zeros(3)
n = 0
for p in Path('VOCdevkit/VOC2012/JPEGImages').glob('*.jpg'):
    img = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    mean += img.reshape(-1, 3).mean(axis=0)
    std += img.reshape(-1, 3).std(axis=0)
    n += 1
mean /= n
std /= n
print('mean =', mean.tolist())
print('std =', std.tolist())
