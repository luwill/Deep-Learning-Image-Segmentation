# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 1 章 — 代码 1-1 对应的Python版本写法如代码1-2所示。
# 原始位置：book-review/drafts/ch01-v3.md:234
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入 opencv 库
import cv2
import sys

# 读取图片
img = cv2.imread("example.jpg")
if img is None:
    sys.exit("Could not read the image.")

# 显示图片
cv2.imshow("Display window", img)

# 保存图片为 png 格式
cv2.imwrite("example.png", img)
