# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:570
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
from monai.metrics import (
    HausdorffDistanceMetric, SurfaceDistanceMetric,
)

# HD95
hd95_metric = HausdorffDistanceMetric(
    include_background=False,
    distance_metric="euclidean",
    percentile=95,         # ← 关键参数
    reduction="mean",
)

# ASD（symmetric=True 即对称版本）
asd_metric = SurfaceDistanceMetric(
    include_background=False,
    symmetric=True,
    reduction="mean",
)

# y_pred / y 形状: (B, C, H, W), 二值或 one-hot
hd95_value = hd95_metric(y_pred=preds, y=labels)
asd_value = asd_metric(y_pred=preds, y=labels)
