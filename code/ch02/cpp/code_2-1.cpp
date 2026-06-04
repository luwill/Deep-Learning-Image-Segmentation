// -*- coding: utf-8 -*-
// 来源：《深度学习图像分割》第 2 章 — 代码 2-1 基于Canny算子的边缘检测算法
// 原始位置：book-review/drafts/ch02-v3.md:262
// 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
// 修改请回写源稿后重新生成，避免代码与书稿失同步。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像：将图像读取为灰度图
    Mat img = imread("./example.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Failed to read image." << endl;
        return -1;
    }

    // 第一步: 高斯平滑
    Mat blurred;
    // 高斯模糊，去除噪声
    GaussianBlur(img, blurred, Size(5, 5), 1.0);

    // 第二步: 使用Canny算子进行边缘检测
    Mat edges;
    double lower_threshold = 50;   // 设置Canny算子的低阈值
    double upper_threshold = 150;  // 设置Canny算子的高阈值
    Canny(blurred, edges, lower_threshold, upper_threshold);

    // 第三步: 保存分割结果
    imwrite("./canny_edges.png", edges);

    // 显示结果
    namedWindow("Canny Edges", WINDOW_NORMAL);
    imshow("Canny Edges", edges);
    waitKey(0);
    return 0;
}
