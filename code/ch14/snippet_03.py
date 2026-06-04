# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:397
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入数据集相关 API
from pathlib import Path
from monai.apps import DecathlonDataset
from monai.data import CacheDataset, SmartCacheDataset, load_decathlon_datalist

# === BraTS ===
train_transform, val_transform = get_transform("brats")

# 定义 BraTS 训练集（DecathlonDataset 内部会自动管理 datalist）
train_dst = DecathlonDataset(
    root_dir=opts.data_root,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    cache_rate=0.0,         # 388 例多模态 4 通道全量缓存约 12 GB，按显存预算调整
    num_workers=4,
)

# 定义 BraTS 验证集
val_dst = DecathlonDataset(
    root_dir=opts.data_root,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=2,
)

# === BTCV ===
train_transforms, val_transforms = get_transform("btcv")

# CODE-12：路径拼接用 pathlib，避免 "./datasets/data" + "btcv.json" 缺斜杠
datasets_json = str(Path(opts.data_root) / "btcv.json")

# 导入数据集
datalist = load_decathlon_datalist(datasets_json, True, "training")
val_files = load_decathlon_datalist(datasets_json, True, "validation")

# 定义 BTCV 训练集（30 例小数据集，全量缓存最高效）
train_dst = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_rate=1.0,         # 全量缓存；cache_num 不再显式指定（避免与 cache_rate 双参冲突）
    num_workers=8,
)

# 定义 BTCV 验证集
val_dst = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=4,
)
