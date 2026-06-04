# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 11 章 — (无标号片段)
# 原始位置：book-review/drafts/ch11-v3.md:325
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
for f in *.json; do labelme_json_to_dataset "$f"; done
