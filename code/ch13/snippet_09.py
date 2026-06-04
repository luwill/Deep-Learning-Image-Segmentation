# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:713
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-4 主函数中模型选项和损失函数选项
# 模型选项（统一使用关键字传参 + opts.num_classes，CODE-M4）
model_map = {
    'unet': models.UNet(num_classes=opts.num_classes),
    'res_unet': models.res_unet34(
        in_channel=3,
        out_channel=opts.num_classes,
        pretrain=True,
    ),
    'nested_unet': models.NestedUNet(
        num_classes=opts.num_classes,
        input_channels=3,
        deep_supervision=False,
    ),
    'unext': models.UNext(num_classes=opts.num_classes),
    'cenet': models.CENet(num_classes=opts.num_classes),
}

# 损失函数选项
if opts.loss_type == 'cross_entropy':
    # CODE-H1：pos_weight 不再硬编码 10，从训练集统计
    pos_weight_value = compute_pos_weight(
        sorted(glob(os.path.join(opts.data_root, 'train_masks', '*.png')))
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value]).to(device)
    )
elif opts.loss_type == 'dice_loss':
    criterion = DiceLoss(smooth=1e-5)
elif opts.loss_type == 'focal_dice':
    criterion = Focal_Dice_Loss()
elif opts.loss_type == 'bce_dice':
    criterion = BCE_Dice_Loss()
