# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:118
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-1 ICH 的 Albumentations 增强
import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import (
    CenterCrop, RandomRotate90, GridDistortion,
    HorizontalFlip, VerticalFlip,
)


def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512
    for x_path, y_path in tqdm(zip(images, masks), total=len(images)):
        x_path = x_path.replace("\\", '/')
        image_name, image_extn = os.path.splitext(
            os.path.basename(x_path))           # CODE-L2 修正：os.path.splitext

        y_path = y_path.replace("\\", '/')
        mask_name, mask_extn = os.path.splitext(
            os.path.basename(y_path))

        # 读取图像（彩色 3 通道）和 mask（严格单通道，CODE-C2）
        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)

        if augment:
            # 中心裁剪：image 与 mask 必须在同一次 aug 调用中完成（CODE-C1）
            aug = CenterCrop(H, W, p=1.0)(image=x, mask=y)
            x1, y1 = aug["image"], aug["mask"]
            # 随机旋转 90 度
            aug = RandomRotate90(p=1.0)(image=x, mask=y)
            x2, y2 = aug["image"], aug["mask"]
            # 网格失真
            aug = GridDistortion(p=1.0)(image=x, mask=y)
            x3, y3 = aug["image"], aug["mask"]
            # 水平翻转
            aug = HorizontalFlip(p=1.0)(image=x, mask=y)
            x4, y4 = aug["image"], aug["mask"]
            # 垂直翻转
            aug = VerticalFlip(p=1.0)(image=x, mask=y)
            x5, y5 = aug["image"], aug["mask"]

            save_images = [x, x1, x2, x3, x4, x5]
            save_masks = [y, y1, y2, y3, y4, y5]
        else:
            save_images = [x]
            save_masks = [y]

        # 保存增强后的图像与掩码（mask 用最近邻 resize，避免插值产生中间灰度，CODE-M1）
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            tmp_img_name = f"{image_name}_{idx}{image_extn}"
            tmp_mask_name = f"{mask_name}_{idx}{mask_extn}"
            cv2.imwrite(os.path.join(save_path, "images", tmp_img_name), i)
            cv2.imwrite(os.path.join(save_path, "masks", tmp_mask_name), m)
            idx += 1
