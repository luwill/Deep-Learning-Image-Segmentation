# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:883
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入相关模块
import numpy as np
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from monai.apps import DecathlonDataset
from datasets.dataset import get_transform
from models.models import get_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CODE-24：root_dir 用 opts.data_root，避免 hard-coded "../"
_, val_transform = get_transform("brats")
val_dst = DecathlonDataset(
    root_dir=opts.data_root,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=1,
)

# 模型映射，略
model_map = ...
model = model_map[opts.model].to(device)

# CODE-25：兼容 model_state / state_dict / 直接 state_dict 三种 checkpoint 结构
ckpt = torch.load(opts.model_root, map_location="cpu")
state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()
print('model loaded.')

# CODE-14：模型预测结果后处理 —— 按数据集区分
if opts.dataset == "brats":
    # BraTS 多标签：sigmoid + 阈值
    post_trans = Compose([Activations(sigmoid=True),
                          AsDiscrete(threshold=0.5)])
else:  # btcv 多类：softmax + argmax + onehot
    post_trans = Compose([Activations(softmax=True),
                          AsDiscrete(argmax=True, to_onehot=opts.out_channels)])

# CODE-5：显式将 autocast 包到 sliding_window_inference 调用处
# 注意：sliding_window_inference 不原生支持 val_amp 参数，此前版本用 inference()
# 自定义包装器，本版改为直接显式 wrap，更符合 MONAI 当前 API
with torch.no_grad():
    sample = val_dst[opts.data_idx]
    test_input = sample["image"].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        test_output = sliding_window_inference(
            inputs=test_input,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
    # 取 batch 第 0 个，再做后处理
    test_output = post_trans(test_output[0])  # [C, D, H, W]

# CODE-15：从输入 MetaTensor 读真实 affine（保留原始 spacing/orientation）
affine = sample["image"].meta["affine"].numpy() \
         if hasattr(sample["image"], "meta") else np.eye(4)

# CODE-16：BraTS 三通道（TC/WT/ET）合并为 0/1/2/4 单通道标签图（BraTS 官方提交格式）
if opts.dataset == "brats":
    pred_np = test_output.cpu().numpy()         # [3, D, H, W]
    seg = np.zeros(pred_np.shape[1:], dtype=np.uint8)
    seg[pred_np[0] == 1] = 1                    # ET（增强肿瘤）→ 1
    seg[pred_np[1] == 1] = 2                    # TC（含 ET）   → 2
    seg[pred_np[2] == 1] = 4                    # WT（含全部） → 4
    nii_img = nib.Nifti1Image(seg, affine=affine)
else:  # btcv：argmax 后已是单通道整数标签
    seg = test_output.argmax(dim=0).cpu().numpy().astype(np.uint8)
    nii_img = nib.Nifti1Image(seg, affine=affine)
nib.save(nii_img, "test_seg.nii.gz")

# 提示：更稳妥的做法是用 MONAI 的 SaveImaged transform 或 Invertd
# 把推理结果反向 transform 回原始空间后保存，可保证 spacing/orientation/origin 完全对齐。
