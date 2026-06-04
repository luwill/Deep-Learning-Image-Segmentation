# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:184
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 错误写法：image 与 mask 解耦——两次调用产生独立随机参数，
#         图像和标注会错位，等价于训练数据被破坏
x_aug = RandomRotate90(p=1.0)(image=x, mask=y)["image"]
y_aug = RandomRotate90(p=1.0)(image=x, mask=y)["mask"]
