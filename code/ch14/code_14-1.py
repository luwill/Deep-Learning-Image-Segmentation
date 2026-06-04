# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — 代码 14-1 是基于MONAI核心库的一个简单范例，旨在让读者对MONAI常用API有一个初步的了解。
# 原始位置：book-review/drafts/ch14-v4.md:96
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 注：以下代码片段改编自 MONAI 官方教程，仅用于演示常用 API 调用形态；
# 其中 ExampleDataset 是占位类（读者实际使用时应替换为自定义 Dataset
# 或 monai.data.CacheDataset / DecathlonDataset 等具体实现，
# 完整的 Dataset 用法参见 14.3.2 节）。

# 导入相关 API
import numpy as np
import torch
from monai.data import DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandRotate,
    ScaleIntensity,
)

# 定义训练集增强方法
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        # …（更多 transforms 略）
    ]
)

# 定义和导入数据集（ExampleDataset 为占位类）
train_ds = ExampleDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size=300, shuffle=True)

# 设置相关训练组件和指定模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1,
                    out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 4
val_interval = 1
auc_metric = ROCAUCMetric()

# 执行训练
for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # …
