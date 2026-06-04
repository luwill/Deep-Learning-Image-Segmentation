# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:422
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# <!-- EDIT-V4: CODE-C1 + CODE-C2 + CODE-H2 + CODE-H3
#               将 validation / scheduler.step / 退出条件从 batch 内层抽到外层；
#               补 device 自适应 + zero_grad(set_to_none=True) + 验证后 model.train()。 -->

# 设备自适应——避免在代码块内出现裸用 'cuda' 导致的 NameError / 无 GPU 拷贝失败
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.train()

# 初始化区间损失
interval_loss = 0

while True:
    # ===== 训练阶段 =====
    cur_epochs += 1
    for (images, labels) in train_loader:
        cur_itrs += 1
        # ImageNet 预训练权重已是 float32，无需手动强转 dtype；
        # AMP（见 12.10.1 节）会自动管理。non_blocking 仅与 pin_memory 配合时生效。
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # PyTorch 1.7+ 推荐用法
        outputs = model(images)
        loss = criterion(outputs, labels)      # CrossEntropyLoss(ignore_index=255)
        loss.backward()
        optimizer.step()
        # PolyLR 配 total_itrs，每 iter 调一次（DeepLab 系列）；
        # 若改用 CosineAnnealingLR / StepLR 等 epoch 维度调度器，则把 step 移到 epoch for 外。
        scheduler.step()

        np_loss = loss.detach().cpu().numpy()
        interval_loss += np_loss
        if vis is not None:
            vis.vis_scalar('Loss', cur_itrs, np_loss)

        # 打印训练信息（按 print_interval 区间平均）
        if cur_itrs % opts.print_interval == 0:
            interval_loss /= opts.print_interval
            print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                  f"Loss={interval_loss:.4f}")
            interval_loss = 0

        # ===== 周期性保存与验证 =====
        if cur_itrs % opts.val_interval == 0:
            # 保存最新检查点（含 model + optimizer + scheduler + epoch + best_score）
            save_ckpt(model, optimizer, scheduler,
                      cur_itrs, best_score,
                      os.path.join(save_path_checkpoints, 'latest.pth'))
            logger.info("Save the latest model to %s" % save_path_checkpoints)

            print("validation...")
            model.eval()
            val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader,
                device=device, metrics=metrics,
                ret_samples_ids=vis_sample_id,
            )
            logger.info("Validation performance: %s", val_score)

            # 保存最优模型（以 mIoU 为最优指标，符合 PASCAL VOC 评测规范）
            if val_score['Mean IoU'] > best_score:
                best_score = val_score['Mean IoU']
                save_ckpt(model, optimizer, scheduler, cur_itrs, best_score,
                          os.path.join(
                              save_path_checkpoints,
                              f'best_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth',
                          ))
                logger.info("Save best-performance model so far to %s"
                            % save_path_checkpoints)

            # 训练过程可视化
            if vis is not None:
                vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    concat_img = np.concatenate((img, target, lbl), axis=2)
                    vis.vis_image('Sample %d' % k, concat_img)

            # 验证完毕显式切回训练态——否则 BN 用 running mean/var、Dropout 失效，
            # 下一个 batch 的训练会显著退化（CODE-C2）。
            model.train()

        # 显式退出条件——达到目标迭代数即结束训练
        if cur_itrs >= opts.total_itrs:
            return
