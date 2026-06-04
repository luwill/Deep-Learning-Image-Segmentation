# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:1237
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-16 执行编译
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
cmake --build . --config Release

# 执行编译后的文件即可得到输出
./deployment ../res_unet34_scripted.pt ../ich_test.jpg
