# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 1 章 — (无标号片段)
# 原始位置：book-review/drafts/ch01-v3.md:265
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入 torch.nn 相关模块
import torch.nn as nn
import torch.nn.functional as F

# 搭建一个 LeNet5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层 2
        self.fc2 = nn.Linear(120, 84)
        # 全连接层 3
        self.fc3 = nn.Linear(84, 10)

    # 定义网络的前向推理流程
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
