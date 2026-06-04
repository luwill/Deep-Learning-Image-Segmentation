# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — 代码 12-6 使用的 visdom 在 2024+ 已较为冷门，主流的训练可视化方案是：
# 原始位置：book-review/drafts/ch12-v4.md:984
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# TensorBoard 集成
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/exp1')
writer.add_scalar('train/loss', np_loss, cur_itrs)
writer.add_scalar('val/mIoU', val_score['Mean IoU'], cur_itrs)
writer.add_image('val/sample', concat_img, cur_itrs)
