# 第 1 章 — 预备知识 · 配套代码

> 抽取自《深度学习图像分割》第 1 章修订稿 `book-review/drafts/ch01-v3.md`。
> 抽取脚本：`book-review/scripts/extract_code.py`。
> 验证脚本：`book-review/scripts/validate_code.py`。

## 概览

Python/C++ 开发环境配置与 OpenCV/PyTorch 基础调用示例。

## 依赖

- python>=3.10
- torch>=2.0
- torchvision

## 文件清单

| 文件 | 状态 | 备注 |
| --- | --- | --- |
| `code_1-1.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `snippet_02.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `cpp/snippet_01.cpp` | 需 OpenCV 4.x | 需 OpenCV / LibTorch 头文件方可编译 |

## 状态说明

- **可独立运行**：单文件 `python file.py` 通过语法校验且无 undefined name。实际跑通仍可能依赖外部数据/模型权重。
- **教学片段**：源自书稿讲解，依赖书稿上下文中已 `import` 或定义的模块/类。可作为参考阅读，需结合书稿前后文方能运行。
- **语法骨架**：源稿用 `def f(self, ...):` 等 Ellipsis 占位、或在 module top-level 出现 `return` 等纯教学截断写法，本仓库保持与书稿一致，**不修改源码**；标注此状态以提示读者。
- **需 OpenCV 4.x / LibTorch**：C++ 示例，依赖外部 SDK 才能编译。

详细 fenced block 盘点见 [`inventory.md`](inventory.md)。
