// -*- coding: utf-8 -*-
// 来源：《深度学习图像分割》第 13 章 — (无标号片段)
// 原始位置：book-review/drafts/ch13-v4.md:1030
// 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
// 修改请回写源稿后重新生成，避免代码与书稿失同步。
// 代码 13-9 cpp 推理脚本（ResNet18 分类示例）
#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    std::cout << "ok\n";
    return 0;
}
