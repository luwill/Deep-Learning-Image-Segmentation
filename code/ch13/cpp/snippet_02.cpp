// -*- coding: utf-8 -*-
// 来源：《深度学习图像分割》第 13 章 — (无标号片段)
// 原始位置：book-review/drafts/ch13-v4.md:1133
// 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
// 修改请回写源稿后重新生成，避免代码与书稿失同步。
// 代码 13-14 deployment.cpp 部署脚本
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: deployment <model_path> <image_path>\n";
        return -1;
    }

    // 1. 导入模型文件
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load the model: " << e.what() << "\n";
        return -1;
    }
    module.eval();

    // 2. 基于 OpenCV 读取影像（默认 BGR 通道）
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << argv[2] << "\n";
        return -1;
    }

    // 3. 与训练端预处理严格对齐：BGR→RGB → resize → /255 → ImageNet mean/std
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);                              // ① BGR→RGB
    cv::resize(img, img, cv::Size(512, 512), cv::INTER_LINEAR);             // ② resize（uint8 上做）
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);                             // ③ 0–1
    cv::Scalar mean(0.485, 0.456, 0.406), stdv(0.229, 0.224, 0.225);
    img -= mean;                                                             // ④ 减均值
    cv::divide(img, stdv, img);                                              // ⑤ 除方差

    // 4. 转换为 torch tensor（显式 dtype + clone 脱离 cv::Mat 生命周期）
    auto img_tensor = torch::from_blob(
        img.data,
        {1, img.rows, img.cols, 3},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();   // NHWC → NCHW

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);

    // 5. 执行预测（推荐用 try/catch 包裹，避免临床部署时崩溃上层 GUI）
    torch::Tensor logits;
    try {
        logits = module.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Model forward failed: " << e.what() << "\n";
        return -1;
    }

    // 6. 后处理：sigmoid + 阈值化为 0/255 二值掩码
    //    （切勿用 cv::normalize 0–255 线性拉伸——会在健康样本上产生伪病灶）
    torch::Tensor probs = torch::sigmoid(logits);
    torch::Tensor mask = (probs > 0.5).to(torch::kU8) * 255;
    mask = mask.squeeze().detach().cpu().contiguous();

    cv::Mat out_img(
        mask.size(0), mask.size(1), CV_8U,
        mask.data_ptr<uint8_t>()
    );
    out_img = out_img.clone();        // 显式拷贝，脱离 mask tensor 生命周期

    // 7. 保存分割图像
    cv::imwrite("./output_image.jpg", out_img);
    std::cout << "ok\n";
    return 0;
}
