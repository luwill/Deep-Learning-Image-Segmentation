# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:353
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import argparse

parser = argparse.ArgumentParser()

# 数据相关参数
parser.add_argument("--data_root", type=str, default='./dataset',
                    help="path to Dataset")
parser.add_argument("--save_root", type=str, default='./',
                    help="path to save result")
parser.add_argument("--dataset", type=str, default='voc',
                    choices=['voc', 'cityscapes', 'ade'],
                    help='Name of dataset')
parser.add_argument("--num_classes", type=int, default=None,
                    help="num classes (default: None)")
