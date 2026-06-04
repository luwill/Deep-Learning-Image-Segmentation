# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:959
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# main.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # cfg.dataset.name, cfg.model.encoder, cfg.optimizer.lr ...
    train(cfg)
