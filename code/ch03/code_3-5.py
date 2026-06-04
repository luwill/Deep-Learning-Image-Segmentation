# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 3 章 — 代码 3-5 深度监督方法示例
# 原始位置：book-review/drafts/ch03-v3.md:483
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 定义一个深度监督类
class DeepSup(nn.Module):
    def __init__(self, num_class=100, fc_dim=2048, use_softmax=False):
        super(DeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        # 定义深监督函数，由卷积、BN 和 ReLU 激活构成
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2,
                                           fc_dim // 4, 1)
        # 最后一层卷积
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4,
                                           num_class, 1, 1, 0)

    # 前向计算流程
    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        # 深度监督模块
        conv4 = conv_out[-2]
        dsv = self.cbr_deepsup(conv4)
        dsv = self.conv_last_deepsup(dsv)
        # 主干卷积网络 softmax 输出
        x = nn.functional.log_softmax(x, dim=1)
        # 深监督分支网络 softmax 输出
        dsv = nn.functional.log_softmax(dsv, dim=1)
        return (x, dsv)
