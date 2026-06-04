# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:548
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# AMP 完整训练循环示例（教学层面，与代码 12-6 的最小循环互补）
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')      # BF16 时可不使用 scaler，但启用 BF16 仍兼容
accum_steps = 1                  # 梯度累积步数；显存紧张时可调到 2 或 4

model.train()
for cur_itrs, (images, labels) in enumerate(train_loader, 1):
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, dtype=torch.long, non_blocking=True)

    # BF16 在 Ampere（A100）及更新的 NVIDIA GPU 上数值范围更宽、不易 NaN；
    # FP16 仅在显存极度紧张且硬件不支持 BF16（V100 / T4）时使用。
    with autocast('cuda', dtype=torch.bfloat16):
        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps

    scaler.scale(loss).backward()
    if cur_itrs % accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()         # PolyLR 每 iter 调度
