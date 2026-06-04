# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:639
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入 argparse 库
import argparse

# 定义配置管理函数，获取相关配置
def get_argparser():
    parser = argparse.ArgumentParser()

    # 数据选项配置
    parser.add_argument("--data_root", type=str,
                        default="./datasets/data",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='brats',
                        choices=['brats', 'btcv'])

    # 模型选项配置
    parser.add_argument("--model", type=str, default='vnet',
                        choices=['vnet', 'segresnet', 'unetr', 'swin_unetr'],
                        help='model name')
    parser.add_argument("--in_channels", type=int, default=4,
                        help="BraTS=4, BTCV=1")
    parser.add_argument("--out_channels", type=int, default=3,
                        help="BraTS=3 (TC/WT/ET), BTCV=14 (13 organs + bg)")
    parser.add_argument("--img_size", type=int, default=96,
                        help="UNETR input cube size")
    parser.add_argument("--feature_size", type=int, default=16,
                        help="UNETR feature_size; SwinUNETR uses 48 internally")

    # 训练选项配置
    parser.add_argument("--epochs", type=int, default=100,
                        help="epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    # CODE-13：DataLoader batch_size 默认值（BraTS 显存大用 2，BTCV 96³ 用 1-2 即可）
    parser.add_argument("--batch_size", type=int, default=2)
    # CODE-6：损失类型选项
    parser.add_argument("--loss_type", type=str, default="dice_ce",
                        choices=["dice", "dice_ce", "dice_focal"],
                        help="loss function variant")

    # 推理选项配置
    parser.add_argument("--roi_size", type=int, default=96,
                        help='validation and inference roi size')
    parser.add_argument("--data_idx", type=int, default=0,
                        help="index of sample for inference")
    parser.add_argument("--model_root", type=str,
                        default="./checkpoints/best.pth")

    # Visdom 可视化选项配置
    parser.add_argument("--enable_vis", action='store_true',
                        default=False)
    # …

    return parser
