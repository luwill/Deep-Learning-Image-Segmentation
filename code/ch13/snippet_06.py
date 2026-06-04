# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:468
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-3 模型评估函数编写
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class StreamSegMetrics(_StreamMetrics):
    """模型评估计算类，包括 Dice、IoU 和 HD 三个指标的计算。"""

    def __init__(self):
        self.dice = []
        self.iou = []
        self.hd = []

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.dice.append(self.dice_coefficient(lt, lp))
            self.iou.append(self.iou_score(lt, lp))
            self.hd.append(self.hausdorff_distance(lt, lp))

    def dice_coefficient(self, y_true, y_pred, smooth=1e-5):
        """二值 Dice，smooth 取 1e-5 避免抬高小目标分数（CODE-C5）。"""
        y_true_f = y_true.flatten().astype(np.float32)
        y_pred_f = y_pred.flatten().astype(np.float32)
        # 双空：健康样本 + 模型未误检 → 完美预测
        if y_true_f.sum() == 0 and y_pred_f.sum() == 0:
            return 1.0
        # 单空：漏检或误检 → 0
        if y_true_f.sum() == 0 or y_pred_f.sum() == 0:
            return 0.0
        intersection = (y_true_f * y_pred_f).sum()
        return (2.0 * intersection + smooth) / (
            y_true_f.sum() + y_pred_f.sum() + smooth
        )

    def iou_score(self, y_true, y_pred, smooth=1e-5):
        y_true_f = y_true.flatten().astype(np.float32)
        y_pred_f = y_pred.flatten().astype(np.float32)
        if y_true_f.sum() == 0 and y_pred_f.sum() == 0:
            return 1.0
        if y_true_f.sum() == 0 or y_pred_f.sum() == 0:
            return 0.0
        intersection = (y_true_f * y_pred_f).sum()
        union = y_true_f.sum() + y_pred_f.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def hausdorff_distance(self, y_true, y_pred):
        """从二值 mask 抽取前景点集后计算对称 HD（CODE-C4）。"""
        # squeeze 掉可能的 channel 维，确保是 (H, W)
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        true_pts = np.argwhere(y_true > 0.5)   # (N, 2) 像素坐标
        pred_pts = np.argwhere(y_pred > 0.5)
        # 双空：健康样本预测正确，HD 定义上为 0
        if len(true_pts) == 0 and len(pred_pts) == 0:
            return 0.0
        # 单空：HD 在数学上无定义，返回 NaN 由 nanmean 跳过
        if len(true_pts) == 0 or len(pred_pts) == 0:
            return np.nan
        d_ab = directed_hausdorff(true_pts, pred_pts)[0]
        d_ba = directed_hausdorff(pred_pts, true_pts)[0]
        return max(d_ab, d_ba)

    def get_results(self):
        # 用 nanmean 跳过 HD 在单空 mask 上的 NaN（CODE-M2）
        return {
            "mean Dice": float(np.nanmean(self.dice)),
            "mIoU": float(np.nanmean(self.iou)),
            "mean HD": float(np.nanmean(self.hd)),
        }
