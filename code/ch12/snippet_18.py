# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:878
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# PyTorch 2.x 训练加速：一行启用
model = model.to(device)
model = torch.compile(model, mode='reduce-overhead')  # 或 'max-autotune'

# AMP 混合精度（与 torch.compile 兼容）
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')

for images, labels in train_loader:
    images = images.to(device); labels = labels.to(device)
    optimizer.zero_grad()
    with autocast('cuda', dtype=torch.float16):
        outputs = model(images)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
