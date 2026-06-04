# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:231
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入相关 API
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
)

# 定义数据转换函数
def get_transform(dataset):
    # ============== BraTS（多模态 MR，3 类多标签：TC/WT/ET）==============
    if dataset == 'brats':
        # 训练集 transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                # CODE-9：BraTS 4 类标签 → TC/WT/ET 3 个二值通道（与 sigmoid + to_onehot_y=False 损失配合）
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # 体素间距统一（image 双线性、label 最近邻；CODE-10）
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # CODE-11：BraTS 多模态 MR 强度归一化的标准做法
                # nonzero=True：背景大量 0 体素，仅在非零区域统计 mean/std
                # channel_wise=True：FLAIR/T1/T1c/T2 4 个模态各自独立 z-score
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                # 随机空间裁剪（BraTS 教程惯用 128³，比 BTCV 96³ 更大）
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(128, 128, 128),
                    random_size=False,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                # 随机强度扰动（BraTS 教程标配）
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        # 验证集 transforms（无随机增广）
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    # ============== BTCV（单模态 CT，14 类多分类：13 器官 + 背景）==============
    if dataset == 'btcv':
        # 训练集 transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # CODE-10：image 双线性、label 最近邻（必须用元组而非单字符串
                # 否则 label 被插值成非整数 ID，DiceLoss 直接错乱）
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                # CT 有 HU 物理单位，可固定窗位窗宽（腹部软组织窗）
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0,
                    clip=True,
                ),
                # 沿前景裁剪降低背景体素占比；MONAI 1.3+ 默认 allow_smaller=False
                # 必须显式传 True 以兼容小体积样本
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    allow_smaller=True,
                ),
                # 关键：基于阳/阴样本均衡的随机裁剪，是 BTCV 训练能收敛的核心 transform
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1, neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.10, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.10, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.10, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
                RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            ]
        )
        # 验证集 transforms（无随机增广，但保留前景裁剪与归一化）
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    allow_smaller=True,
                ),
            ]
        )

    return train_transforms, val_transforms
