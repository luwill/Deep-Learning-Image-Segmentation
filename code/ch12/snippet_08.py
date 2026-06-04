# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:311
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for dense prediction. Lin et al., ICCV 2017.

    多类语义分割版本：alpha 既可为标量（教学示例的简化形式），
    也可为 shape=[C] 的 tensor 表达类别权重；ignore_index 用于跳过
    VOC 中值为 255 的边界 / 无关像素。
    """

    # <!-- EDIT-V4: CODE-C5 FocalLoss 加 ignore_index 与有效像素归一化 -->
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 标准交叉熵——显式传 ignore_index，把 255 像素的 ce_loss 置 0
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            ignore_index=self.ignore_index,
        )
        # Focal 调制项
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # 仅按"有效像素"求平均，避免 .mean() 把 ignore 位置当成分母
        valid = (targets != self.ignore_index)
        return focal_loss.sum() / valid.sum().clamp(min=1)
