# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:326
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import re, json, random, os

def patient_level_split(image_dir, val_ratio=0.2, seed=42):
    """按 patient_id 划分 train/val，返回两个 patient_id 集合。"""
    pattern = re.compile(r'(\d+)-slice')
    patient_ids = sorted({
        pattern.match(f).group(1)
        for f in os.listdir(image_dir)
        if pattern.match(f) is not None
    })
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * (1 - val_ratio))
    train_ids = set(patient_ids[:split_idx])
    val_ids = set(patient_ids[split_idx:])
    return train_ids, val_ids


def filter_by_patient(image_dir, patient_id_set):
    """根据 patient_id 集合过滤切片文件名。"""
    pattern = re.compile(r'(\d+)-slice')
    return sorted([
        f for f in os.listdir(image_dir)
        if pattern.match(f) and pattern.match(f).group(1) in patient_id_set
    ])


# 用法
train_ids, val_ids = patient_level_split('data/ich/all_imgs', seed=42)
# 持久化到 splits/ich_split.json，确保多次实验可复现
with open('splits/ich_split.json', 'w') as f:
    json.dump(
        {'train': sorted(train_ids), 'val': sorted(val_ids)},
        f, indent=2,
    )
