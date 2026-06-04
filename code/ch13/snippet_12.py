# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:966
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-7 基于 Script 的 PyTorch 模型表示
my_model = MyModel()        # 实例化
my_model.eval()             # 切到 eval 模式（关闭 BN/Dropout 训练行为）
scripted_model = torch.jit.script(my_model)   # ← 传入实例，不是类
scripted_model.save('model.pt')
