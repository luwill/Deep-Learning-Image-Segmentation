// -*- coding: utf-8 -*-
// 来源：《深度学习图像分割》第 1 章 — (无标号片段)
// 原始位置：book-review/drafts/ch01-v3.md:203
// 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
// 修改请回写源稿后重新生成，避免代码与书稿失同步。
// 包含 OpenCV 的头文件
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    // 读取图像
    cv::Mat img = cv::imread("example.jpg");
    // 检查图像是否加载成功
    if (img.empty())
    {
        std::cout << "图像加载失败，请检查路径。" << std::endl;
        return -1;
    }
    // 创建窗口
    cv::namedWindow("显示图像", cv::WINDOW_AUTOSIZE);
    // 在窗口中展示图像
    cv::imshow("显示图像", img);
    // 等待用户按下任意键后关闭窗口
    cv::waitKey(0);
    // 另存为 PNG 格式
    cv::imwrite("example.png", img);
    return 0;
}
