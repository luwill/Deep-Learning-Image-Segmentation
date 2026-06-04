# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:1001
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-8 PyTorch 模型的 TorchScript 转换
import torch
from torchvision.models import resnet18
from torchvision import transforms as T
from PIL import Image

model = resnet18(weights="DEFAULT")     # 真实预训练权重
model.eval()

# Dummy input 应尽量贴近真实分布——用真实预处理后的张量，
# 而非 torch.rand（[0,1] 均匀分布与训练分布失配）
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
example = preprocess(Image.open('sample.jpg').convert('RGB')).unsqueeze(0)

with torch.no_grad():
    traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("traced_resnet_model.pt")
