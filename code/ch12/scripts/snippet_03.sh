# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:970
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 命令行任意覆盖
python main.py optimizer.lr=0.01 model.encoder=resnet50
# 多任务扫描
python main.py -m optimizer.lr=0.01,0.005,0.001 model.encoder=resnet50,resnet101
