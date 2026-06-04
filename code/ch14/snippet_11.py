# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:1337
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 旧写法（PyTorch 1.x，仍可用但有 deprecation warning）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()

# 新写法（PyTorch 2.x 推荐，需 PyTorch >= 2.3）
scaler = torch.amp.GradScaler('cuda')
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
