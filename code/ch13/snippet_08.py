# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:613
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import re
from collections import defaultdict

def aggregate_dice_per_patient(images, preds, labels):
    """images: 切片文件名列表；preds/labels: 与之等长的二值 mask 列表。"""
    pattern = re.compile(r'(\d+)-slice')
    inter_per_patient = defaultdict(float)
    union_per_patient = defaultdict(float)
    for img_name, p, l in zip(images, preds, labels):
        pid = pattern.match(img_name).group(1)
        inter_per_patient[pid] += float((p * l).sum())
        union_per_patient[pid] += float(p.sum() + l.sum())
    # 每个患者一个 Dice
    dices = [
        (2 * inter_per_patient[pid] + 1e-5) /
        (union_per_patient[pid] + 1e-5)
        for pid in inter_per_patient
    ]
    return float(np.mean(dices))
