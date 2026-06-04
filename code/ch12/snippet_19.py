# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:904
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# torchvision.transforms.v2 联合变换示例（替代代码 12-2 中自定义的 et.Compose）
from torchvision.transforms import v2 as T
from torchvision import tv_tensors

train_transform = T.Compose([
    T.ToImage(),                            # PIL / ndarray → Image
    T.RandomResizedCrop(513, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# 使用方式（mask 用 tv_tensors.Mask 包裹，v2 自动联合变换）
img = Image.open(img_path).convert('RGB')
mask = tv_tensors.Mask(np.array(Image.open(mask_path)))
img, mask = train_transform(img, mask)
