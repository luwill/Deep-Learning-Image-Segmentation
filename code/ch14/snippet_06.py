# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:709
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 前序代码略
# …

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose

# 定义验证函数（CODE-17 补完整最小可运行体）
def validate(opts, model, loader, device, dataset):
    """对整个 val_loader 跑一遍 sliding_window_inference + DiceMetric。"""
    model.eval()

    # 按数据集区分后处理（CODE-14）
    if dataset == "brats":
        post_pred = Compose([Activations(sigmoid=True),
                             AsDiscrete(threshold=0.5)])
        post_label = lambda x: x  # BraTS label 已经是 3 通道二值图
    else:  # btcv
        post_pred = Compose([Activations(softmax=True),
                             AsDiscrete(argmax=True, to_onehot=opts.out_channels)])
        post_label = Compose([AsDiscrete(to_onehot=opts.out_channels)])

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = sliding_window_inference(
                    inputs=x, roi_size=(opts.roi_size,)*3,
                    sw_batch_size=4, predictor=model,
                    overlap=0.5, mode="gaussian",
                )
            pred = [post_pred(p) for p in pred]
            y = [post_label(yy) for yy in y]
            dice_metric(y_pred=pred, y=y)
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice


# 定义主函数
def main():
    # 加载配置项
    opts = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_dst, val_dst = get_dataset(opts)

    # 数据导入（CODE-13：num_workers=4, pin_memory=True）
    train_loader = DataLoader(
        train_dst, batch_size=opts.batch_size,
        shuffle=True, num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dst, batch_size=1,                  # 验证 batch_size 通常固定 1
        shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 模型部分
    model_map = {
        'vnet': get_models('vnet', in_channels=opts.in_channels,
                           out_channels=opts.out_channels),
        'segresnet': get_models('segresnet',
                                in_channels=opts.in_channels,
                                out_channels=opts.out_channels),
        'unetr': get_models('unetr', in_channels=opts.in_channels,
                            out_channels=opts.out_channels,
                            img_size=opts.img_size,
                            feature_size=opts.feature_size),
        'swin_unetr': get_models('swin_unetr',
                                 in_channels=opts.in_channels,
                                 out_channels=opts.out_channels,
                                 feature_size=48),
    }
    model = model_map[opts.model].to(device)

    # 损失函数（CODE-6：补 DiceCELoss / DiceFocalLoss 分支）
    if opts.loss_type == 'dice':
        if opts.dataset == 'brats':
            criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5,
                                 squared_pred=True,
                                 to_onehot_y=False, sigmoid=True)
        else:
            criterion = DiceLoss(to_onehot_y=True, softmax=True)
    elif opts.loss_type == 'dice_ce':
        if opts.dataset == 'brats':
            criterion = DiceCELoss(sigmoid=True, squared_pred=True,
                                   lambda_dice=1.0, lambda_ce=1.0)
        else:
            criterion = DiceCELoss(to_onehot_y=True, softmax=True,
                                   lambda_dice=1.0, lambda_ce=1.0)
    else:  # dice_focal
        if opts.dataset == 'brats':
            criterion = DiceFocalLoss(sigmoid=True, lambda_dice=1.0)
        else:
            criterion = DiceFocalLoss(to_onehot_y=True, softmax=True,
                                      lambda_dice=1.0)

    # 优化器（4 个网络的推荐 lr/weight_decay 见 14.4 节末对照表）
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr,
                                  weight_decay=1e-5)

    # 调度器：朴素 cosine 衰减；SwinUNETR / UNETR 推荐带 warmup 的版本：
    # from monai.optimizers import LinearWarmupCosineAnnealingLR
    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer, warmup_epochs=opts.epochs // 20, max_epochs=opts.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.epochs)

    # CODE-7：混合精度训练初始化（PyTorch 2.x 推荐写法，旧写法见 14.11 节）
    scaler = torch.amp.GradScaler('cuda')

    # 训练部分
    best_metric = -1.0
    for epoch in range(opts.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()

        # 验证部分（CODE-17：BraTS / BTCV 都走 validate 函数，不再分支）
        if (epoch + 1) % 5 == 0:
            mean_dice = validate(opts, model, val_loader,
                                  device=device, dataset=opts.dataset)
            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, mean Dice={mean_dice:.4f}")
            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save({"model_state": model.state_dict(),
                            "epoch": epoch, "best_metric": best_metric},
                           opts.model_root)
