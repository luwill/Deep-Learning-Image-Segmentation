# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:212
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dst,
    batch_size=opts.batch_size,        # VOC 上 DeepLab v3+ 常用 8 / 16
    shuffle=True,
    num_workers=opts.num_workers,      # 一般 4–8，按 CPU 核数调整
    pin_memory=True,                   # 配合 .to(non_blocking=True) 加速 H2D 拷贝
    persistent_workers=True,           # PyTorch 1.7+，避免每 epoch 重启 worker
    prefetch_factor=2,
    drop_last=True,                    # 避免最后不满 batch 的 BN 抖动
    worker_init_fn=seed_worker,        # 复现性：见 12.6 节 set_seed 配套实现
    generator=g,                       # 同上
)
val_loader = DataLoader(
    val_dst, batch_size=1,             # 验证时建议 batch=1，避免 padding 影响指标
    shuffle=False, num_workers=opts.num_workers,
    pin_memory=True, persistent_workers=True,
)
