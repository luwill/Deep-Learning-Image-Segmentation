# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:1033
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入相关模块
import torch
from models.models import get_models

# 定义模型
model = get_models('unetr', in_channels=1, out_channels=14,
                   img_size=96, feature_size=16)

# 导入模型文件（兼容多种 checkpoint 键名）
ckpt = torch.load('best_unetr_btcv.pth', map_location="cpu")
state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
model.load_state_dict(state_dict)
model.to('cuda')
model.eval()
print("model loaded.")

# 通过 jit.script 进行模型转换
try:
    scripted = torch.jit.script(model)
    scripted.save("models/best_unetr_btcv.ts")
except Exception as e:
    # CODE-26：UNETR 内含 nn.MultiheadAttention 与可变长度 transformer 块，
    # 1.4+ 引入的 Flash Attention 路径含动态分支，部分版本 jit.script 会失败；
    # 此时回退到 jit.trace 是更稳妥的备选方案
    print(f"jit.script failed: {e}; fallback to jit.trace")
    dummy = torch.randn(1, 1, 96, 96, 96, device="cuda")
    traced = torch.jit.trace(model, dummy)
    traced.save("models/best_unetr_btcv.ts")
