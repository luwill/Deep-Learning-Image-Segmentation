# 《深度学习图像分割》配套代码库

> 本仓库是《深度学习图像分割》（louwill 著）一书的配套代码库。
> 全书共 15 章，从传统分割算法、深度学习基础架构，到 2024–2026 年的基础模型 (SAM/SAM 2)、
> 半监督/弱监督/自监督分割，再到医学影像与遥感的工程实战。

## 仓库结构

```
Deep-Learning-Image-Segmentation/
├── README.md                ← 本文件
├── code/                    ← 按章组织的配套代码（本次更新新增）
│   ├── ch01/ … ch14/        ← 每章一个目录，含 README + inventory
│   └── ch15/                ← 总结章（无代码）
├── Scripts/                 ← 早期 v1.0 版本的 C++ 传统算法实现
│   ├── canny.cpp / otsu.cpp / watershed.cpp / region_growth.cpp
│   ├── mh.cpp (Marr-Hildreth) / grabcut.cpp
│   └── 与 code/ch02/cpp/ 内容重合，保留以兼容历史链接。
└── 第1章 预备知识.pdf
```

## 各章代码索引

| 章 | 主题 | Python | C++ | 其他 |
| -- | ---- | ------ | --- | ---- |
| 01 | 预备知识 [ch01/](code/ch01/README.md) | 2 | 1 | — |
| 02 | 传统图像分割算法 [ch02/](code/ch02/README.md) | 0 | 5 | — |
| 03 | 深度学习图像分割基础 [ch03/](code/ch03/README.md) | 4 | 0 | — |
| 04 | FCN 与 U-Net 系列 [ch04/](code/ch04/README.md) | 8 | 0 | — |
| 05 | DeepLab 与编解码结构 [ch05/](code/ch05/README.md) | 2 | 0 | — |
| 06 | 注意力机制与 Transformer 分割 [ch06/](code/ch06/README.md) | 2 | 0 | — |
| 07 | 三维图像分割 [ch07/](code/ch07/README.md) | 1 | 0 | — |
| 08 | 实例分割与全景分割 [ch08/](code/ch08/README.md) | 0 | 0 | — |
| 09 | 基础模型与开放词表分割 [ch09/](code/ch09/README.md) | 0 | 0 | — |
| 10 | 半监督、弱监督与自监督分割 [ch10/](code/ch10/README.md) | 0 | 0 | — |
| 11 | 图像分割数据集与评价指标 [ch11/](code/ch11/README.md) | 0 | 0 | shell×1 |
| 12 | 图像分割工程实战 [ch12/](code/ch12/README.md) | 23 | 0 | yaml×1 / shell×3 |
| 13 | 医学影像分割项目实战 [ch13/](code/ch13/README.md) | 15 | 2 | CMake×2 / shell×3 |
| 14 | 遥感与工业图像分割项目实战 [ch14/](code/ch14/README.md) | 11 | 0 | shell×2 |
| 15 | 全书总结与展望 [ch15/](code/ch15/README.md) | 0 | 0 | — |

## 使用方法

1. 按章 `cd code/chXX/`，阅读该章 `README.md` 了解依赖与文件状态。
2. 状态标注：
   - **可独立运行**：单文件可经 `py_compile` + `pyflakes` 校验。
   - **教学片段**：依赖书稿上下文（已 import 的模块、上文已定义的类）。
   - **语法骨架**：原稿采用 Ellipsis 占位或 module-top-level `return` 的教学截断写法，
     本仓库**忠于书稿、不做修改**，仅在 README 中标注。
3. 对应章节 `inventory.md` 中按源稿行号列出每个 fenced block，便于回查原文。

## 抽取与验证流水线

代码并非手工誊写，而是从 v3/v4 修订稿中以脚本自动抽取，保证与书稿同步：

```
book-review/drafts/chXX-vN.md   ← 修订稿源
        │
        │  book-review/scripts/extract_code.py
        ▼
code/chXX/{*.py, cpp/*.cpp, scripts/*.sh, configs/*.yaml}
        │
        │  book-review/scripts/validate_code.py
        │      ├─ python -m py_compile
        │      └─ python -m pyflakes
        ▼
book-review/scripts/validate_report.md
```

## 已知限制

- **C++ 部分**：本地 CI 未安装 OpenCV/LibTorch，因此 ch02/ch13 的 cpp 文件仅做了静态导出，
  未做 `g++ -fsyntax-only` 通过验证。其内容与源稿逐字一致，编译/运行验证需读者本地具备 SDK。
- **教学片段**：书稿的代码主要服务于讲解，不少是依赖上下文的片段（snippet），并非 standalone script。每章 README 显式区分。
- **大模型权重**：ch12–ch14 的训练 / 推理 / SAM 集成示例需读者自行下载权重，
  本仓库不包含权重文件。

## 配合书稿使用

代码块在书稿中的标号形如「代码 X-Y」。本仓库文件命名优先匹配该标号：

- `code_X-Y.py` / `code_X-Y.cpp` — 书稿明确编号的代码示例。
- `snippet_NN.py` 等 — 书稿正文中没有独立编号的内联示例（按章内序号排）。

每章 `inventory.md` 列出每个 fenced block 的原稿行号，便于精确对应。

## 早期版本兼容

`Scripts/` 目录是仓库 v1.0 时期的 C++ 实现，主要对应第 2 章（传统分割算法）。
新版按章重组后，等价代码在 `code/ch02/cpp/` 下。保留 `Scripts/` 以兼容历史链接。

---

若发现代码与书稿正文不一致，请优先以书稿修订稿（`book-review/drafts/chXX-vN.md`）为准；
并欢迎通过 issue / PR 反馈勘误。
