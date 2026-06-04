# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:123
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入相关模块
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# 定义 VOC 数据集类
class VOCSegmentation(Dataset):
    """A PyTorch Dataset class for the VOC Segmentation dataset."""

    def __init__(self, root, image_set='train', transform=None):
        """
        Args:
            root (str): 数据集根目录
            image_set (str): 'train' 或 'val'
            transform (callable, optional): 同时作用于图像与掩码的变换
        """
        self.root = Path(root)
        self.transform = transform
        self.image_set = image_set

        base_dir = 'VOCdevkit/VOC2012'
        voc_root = self.root / base_dir
        if not voc_root.is_dir():
            raise RuntimeError(f'Dataset not found at {voc_root}.')

        image_dir = voc_root / 'JPEGImages'
        mask_dir = voc_root / 'SegmentationClass'
        splits_dir = voc_root / 'ImageSets/Segmentation'
        split_f = splits_dir / f"{image_set}.txt"

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [image_dir / f"{x}.jpg" for x in file_names]
        self.masks = [mask_dir / f"{x}.png" for x in file_names]
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # PASCAL VOC 分割掩码是 'P' 模式的调色板索引图，类别为 0–20，
        # 边界与无关像素值为 255（即 ignore_index）。
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        else:
            # 当未传入 transform 时，显式把掩码转成 LongTensor，
            # 保留 0–20 类别索引与 255 ignore——避免 transforms.ToTensor()
            # 把 mask 缩放到 [0, 1] float32 而类别全乱。
            # <!-- EDIT-V4: CODE-C4 显式 mask → LongTensor + 保留 255 ignore -->
            target = torch.as_tensor(np.array(target), dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.images)
