# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:376
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 配套 utils/seed.py 或直接放在 main.py 顶部
import os, random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """复现性最小骨架——分割训练前调用一次。

    deterministic=True 时启用 PyTorch 的确定性算法（部分算子会牺牲性能）；
    通常仅在最终发版前的对比实验中开启，平时训练优先使用 cudnn.benchmark
    以让 cuDNN 自动搜索最优卷积算法。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """配合 DataLoader(worker_init_fn=seed_worker, generator=g) 使用。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# DataLoader 端的 generator
g = torch.Generator()
g.manual_seed(42)
