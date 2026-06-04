# 第 13 章 — 医学影像分割项目实战 · 配套代码

> 抽取自《深度学习图像分割》第 13 章修订稿 `book-review/drafts/ch13-v4.md`。
> 抽取脚本：`book-review/scripts/extract_code.py`。
> 验证脚本：`book-review/scripts/validate_code.py`。

## 概览

医学影像分割实战（ICH 出血分割），含 PyTorch 训练与 LibTorch C++ 部署。

## 依赖

- torch>=2.0
- monai
- nibabel（DICOM/NIfTI 读取）
- OpenCV 4.x（C++ 推理示例）
- LibTorch C++（CMake 构建）

## 文件清单

| 文件 | 状态 | 备注 |
| --- | --- | --- |
| `code_13-1.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `snippet_01.py` | 可独立运行 | py_compile 通过；未使用的 import (源稿教学保留) |
| `snippet_02.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_04.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_05.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `snippet_06.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_07.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_08.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_09.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_10.py` | 可独立运行 | py_compile 通过；未使用的 import (源稿教学保留) |
| `snippet_11.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `snippet_12.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `snippet_13.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `snippet_14.py` | 可独立运行 | py_compile + pyflakes 通过 |
| `snippet_15.py` | 教学片段 | 依赖书稿上下文的 import / 上层定义 |
| `cpp/snippet_01.cpp` | 需 OpenCV 4.x | 需 OpenCV / LibTorch 头文件方可编译 |
| `cpp/snippet_02.cpp` | 需 OpenCV 4.x | 需 OpenCV / LibTorch 头文件方可编译 |
| `cpp/CMakeLists_code_13-14.txt` | 构建脚本 | CMake 配置 |
| `cpp/CMakeLists_snippet_01.txt` | 构建脚本 | CMake 配置 |
| `scripts/snippet_01.sh` | 脚本 | 下载 / 安装命令 |
| `scripts/snippet_02.sh` | 脚本 | 下载 / 安装命令 |
| `scripts/snippet_03.sh` | 脚本 | 下载 / 安装命令 |

## 状态说明

- **可独立运行**：单文件 `python file.py` 通过语法校验且无 undefined name。实际跑通仍可能依赖外部数据/模型权重。
- **教学片段**：源自书稿讲解，依赖书稿上下文中已 `import` 或定义的模块/类。可作为参考阅读，需结合书稿前后文方能运行。
- **语法骨架**：源稿用 `def f(self, ...):` 等 Ellipsis 占位、或在 module top-level 出现 `return` 等纯教学截断写法，本仓库保持与书稿一致，**不修改源码**；标注此状态以提示读者。
- **需 OpenCV 4.x / LibTorch**：C++ 示例，依赖外部 SDK 才能编译。

详细 fenced block 盘点见 [`inventory.md`](inventory.md)。
