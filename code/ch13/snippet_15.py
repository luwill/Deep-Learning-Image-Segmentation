# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:1279
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 13.11.5 MedSAM finetune 起手式
import torch
import torch.nn as nn
from segment_anything import sam_model_registry

# 加载 MedSAM ViT-B 权重（HuggingFace 上有镜像）
sam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")

# 冻结 image_encoder 与 prompt_encoder
for param in sam.image_encoder.parameters():
    param.requires_grad = False
for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

# 仅微调 mask_decoder（参数量约 4M，单卡 24G 可轻松训练）
optimizer = torch.optim.AdamW(
    sam.mask_decoder.parameters(),
    lr=1e-4, weight_decay=1e-2,
)

# 损失：DiceLoss + BCEWithLogitsLoss（无显式权重）
dice_loss_fn = DiceLoss(smooth=1e-5)
bce_loss_fn = nn.BCEWithLogitsLoss()

def loss_fn(logits, target):
    return dice_loss_fn(logits, target) + bce_loss_fn(logits, target)

# 输入分辨率 1024×1024（MedSAM 训练分辨率）
# Prompt 用 ground-truth mask 的外接 box（训练时由 GT 自动生成）
