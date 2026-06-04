# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:99
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入数据对象类 Dataset
from torch.utils.data import Dataset

# 基于 Dataset 的数据定义和读取框架
class CustomDataset(Dataset):
    def __init__(self, ...):
        # 初始化：定义数据路径、文件列表、变换方法等
        ...

    def __getitem__(self, index):
        # 单样本读取：按索引读取一对 (img, label)，并应用变换
        ...
        return (img, label)

    def __len__(self):
        # 返回数据集样本数量
        return len(self.images)
