# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:938
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-6 基于 Tracing 的 PyTorch 模型表示
import torch


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


my_model = MyModel()
my_model.eval()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_model = torch.jit.trace(my_model, (x, h))
traced_model.save('model.pt')
