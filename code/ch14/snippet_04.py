# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:484
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入 monai 相关分割模型
from monai.networks.nets import SegResNet, UNETR, VNet, SwinUNETR

# 定义模型获取函数
def get_models(model_name,
               in_channels,
               out_channels,
               img_size=None,
               feature_size=None):
    # SegResNet（CNN 编解码）
    if model_name == 'segresnet':
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,                  # Myronenko 2018 论文取 16；MONAI 默认 8
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0.2,
            # upsample_mode 默认 NONTRAINABLE（trilinear），论文用 pixel-shuffle
            # 如需复刻论文可显式传 upsample_mode="pixelshuffle"
        )

    # VNet（CNN 编解码，3D U-shape）
    if model_name == 'vnet':
        # ⚠️ MONAI 1.3+ 已移除 dropout_prob，改用 dropout_prob_down / dropout_prob_up
        # 1.4+ 下的等价写法见 14.11 节表 14-2；以下为 1.2 版本写法
        model = VNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0.2,                 # 仅 1.2 兼容；1.3+ 报错
            dropout_dim=3,
            # act 默认在 1.3+ 改为 ELU；论文（Milletari 2016）用 PReLU
            # 如需复刻论文可显式传 act=("PRELU", {"num_parameters": 1})
        )

    # UNETR（纯 ViT 编码 + CNN 解码）
    if model_name == 'unetr':
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(img_size, img_size, img_size),
            feature_size=feature_size,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            # CODE-2：1.4+ 已移除 pos_embed，改用 proj_type
            # 注意 proj_type="conv" 是当前 MONAI 默认（patch 卷积）
            # proj_type="perceptron" 才对应 UNETR 论文写法（一个全连接层），二者结构不等价
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            # qkv_bias 默认 False；UNETR 论文图示带 bias，如需复刻可显式 qkv_bias=True
        )

    # SwinUNETR（Swin Transformer 编码 + CNN 解码）
    if model_name == 'swin_unetr':
        # CODE-1：MONAI 1.3.1+ 已移除 img_size 参数（推理时由输入张量动态推断）
        # 1.3+ 还引入 use_v2=True 切换到 Swin V2（双路 + 残差），建议默认开启以提升精度
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,                  # 论文值；MONAI 默认 24
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            use_v2=True,                      # 启用 SwinUNETR-v2（MONAI 1.3+）
        )

    return model
