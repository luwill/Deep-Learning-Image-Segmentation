// -*- coding: utf-8 -*-
// 来源：《深度学习图像分割》第 2 章 — 代码 2-3 区域生长算法
// 原始位置：book-review/drafts/ch02-v3.md:480
// 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
// 修改请回写源稿后重新生成，避免代码与书稿失同步。
#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 定义区域生长分割算法
void regionGrowing(const Mat& src, Mat& dst, Point seed, int threshold) {
    // 图像尺寸
    int rows = src.rows;
    int cols = src.cols;

    // 初始化输出图像，所有像素初始为0（黑色）
    dst = Mat::zeros(src.size(), CV_8UC1);

    // 定义8邻域
    int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
    int dy[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };

    // 获取种子点的灰度值
    int seedGrayValue = src.at<uchar>(seed);

    // 创建一个队列用于存储待处理的点
    queue<Point> pointQueue;
    pointQueue.push(Point(seed.x, seed.y));

    // 将种子点标记为已处理
    dst.at<uchar>(seed) = 255;  // 标记区域为白色

    // 开始区域生长
    while (!pointQueue.empty()) {
        Point currentPoint = pointQueue.front();
        pointQueue.pop();

        // 遍历当前点的8邻域
        for (int i = 0; i < 8; i++) {
            int newX = currentPoint.x + dx[i];
            int newY = currentPoint.y + dy[i];

            // 确保邻域点在图像范围内
            if (newX >= 0 && newX < cols && newY >= 0 && newY < rows) {
                // 如果该点还没有被标记并且与种子点的灰度差小于阈值
                int neighborGrayValue = src.at<uchar>(newY, newX);
                if (dst.at<uchar>(newY, newX) == 0 &&
                    abs(neighborGrayValue - seedGrayValue) <= threshold) {
                    // 将该点加入到队列中
                    pointQueue.push(Point(newX, newY));
                    // 标记该点为区域的一部分
                    dst.at<uchar>(newY, newX) = 255;
                }
            }
        }
    }
}
