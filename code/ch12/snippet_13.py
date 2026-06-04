# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:582
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# <!-- EDIT-V4: CODE-H4 + CODE-M6
#               transform / device 入参化、改用 torch.inference_mode()、删调试 print。 -->
import os
import numpy as np
import torch
from PIL import Image


def inference(model, test_img, transform, device='cuda'):
    """单张图像推理，输出 RGB 三通道伪彩色掩码。

    Args:
        model:     已加载权重并切换到 eval 模式的分割网络。
        test_img:  待推理图像路径。
        transform: 与验证一致的 tensor 化 + 归一化变换。
        device:    'cuda' / 'cpu' / 'mps'。
    """
    img = Image.open(test_img).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    # PyTorch 1.9+ 推荐 inference_mode() 替代 no_grad()——
    # 后者会进一步禁止 view 跟踪，对推理吞吐有约 5%–10% 的提升。
    with torch.inference_mode():
        outputs = model(input_tensor)

    preds = outputs.argmax(dim=1).cpu().numpy()
    pred_rgb = VOCSegmentation.decode_target(preds[0]).astype(np.uint8)
    save_path = f"{test_img.rsplit('.', 1)[0]}_pred.png"
    Image.fromarray(pred_rgb).save(save_path)
    return save_path
