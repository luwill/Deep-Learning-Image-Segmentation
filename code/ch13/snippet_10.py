# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:803
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-5 2D 医学影像的推理
import sys
sys.path.append("..")

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

import models
from datasets import (
    ICHSegmentation, ISICSegmentation, BUSISegmentation,
)


def get_argparser():
    parser = argparse.ArgumentParser()
    # 数据选项
    parser.add_argument(
        "--data_root", type=str, default='', help="path to dataset",
    )
    parser.add_argument(
        "--dataset", type=str, default='ich',
        choices=['ich', 'busi', 'isic'],
        help='dataset name (controls preprocess pipeline)',
    )
    # 模型选项
    parser.add_argument(
        "--model", type=str, default='unet',
        choices=['unet', 'res_unet', 'nested_unet', 'unext', 'cenet'],
        help='model name',
    )
    parser.add_argument(
        "--model_root", type=str, help='path to model file',
    )
    return parser


# CODE-H2：preprocess 按数据集分发，避免所有数据集复用 ICH 静态方法
PREPROCESS_MAP = {
    'ich': ICHSegmentation.image_preprocess,
    'busi': BUSISegmentation.image_preprocess,
    'isic': ISICSegmentation.image_preprocess,
}


def inference(model, test_img, dataset_name):
    img = Image.open(test_img).convert('RGB')
    img = PREPROCESS_MAP[dataset_name](img)                # 按数据集选 preprocess
    img = torch.from_numpy(img).type(torch.float32)
    img = img.unsqueeze(0).to('cuda')

    with torch.no_grad():
        outputs = model(img)

    # CODE-M5：显式 sigmoid + 阈值化，比 np.rint 隐式语义清晰
    probs = torch.sigmoid(outputs).cpu().numpy()
    preds = (probs > 0.5).astype(np.uint8) * 255
    preds = np.squeeze(preds)

    # CODE-M6：用 splitext 而非 split('.')[0]，对路径含点号鲁棒
    base = os.path.splitext(test_img)[0]
    Image.fromarray(preds).save(f'{base}_pred.png')


if __name__ == "__main__":
    opts = get_argparser().parse_args()
    model_map = {
        'unet': models.UNet(num_classes=1),
        'res_unet': models.res_unet34(
            in_channel=3, out_channel=1, pretrain=True,
        ),
        'nested_unet': models.NestedUNet(
            num_classes=1, input_channels=3, deep_supervision=False,
        ),
        'unext': models.UNext(num_classes=1),
        'cenet': models.CENet(num_classes=1),
    }
    model = model_map[opts.model]
    model.load_state_dict(
        torch.load(opts.model_root)['model_state']
    )
    model.to('cuda')
    model.eval()
    print('model loaded.')
    inference(model, opts.data_root, opts.dataset)
