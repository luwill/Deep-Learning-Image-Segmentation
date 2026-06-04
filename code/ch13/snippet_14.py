# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:1104
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-13 Res-UNet 模型的 Script 转换
import torch
from models.res_unet import res_unet34

model = res_unet34(in_channel=3, out_channel=1, pretrain=True)   # 关键字传参
model.load_state_dict(
    torch.load('best_res_unet_ich.pt')['model_state']
)
model.eval()                                                      # ① 切 eval 模式

with torch.no_grad():
    scripted_model = torch.jit.script(model)
scripted_model.save("./res_unet34_scripted.pt")

# ② 数值一致性校验：导出前用同一输入比对 Python 与 Script 输出
example = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    py_out = model(example)
    script_out = scripted_model(example)
assert torch.allclose(py_out, script_out, atol=1e-5), \
    "TorchScript 输出与 Python 输出数值不一致，部署前需排查！"
print("Python vs Script 一致性 OK.")
