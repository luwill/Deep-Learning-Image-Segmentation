#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Canny边缘检测
int main() {
    // 读取图像
    Mat img = imread("./example.png", IMREAD_GRAYSCALE); // 将图像读取为灰度图

    if (img.empty()) {
        cout << "无法读取图片！" << endl;
        return -1;
    }

    // Step 1: 高斯模糊 (平滑处理)
    Mat blurred;
    GaussianBlur(img, blurred, Size(5, 5), 1.0);  // 高斯模糊，去除噪声

    // Step 2: 使用Canny算子进行边缘检测
    Mat edges;
    double lower_threshold = 50;  // 设置Canny算子的低阈值
    double upper_threshold = 150; // 设置Canny算子的高阈值
    Canny(blurred, edges, lower_threshold, upper_threshold);

    // Step 3: 保存分割结果
    imwrite("./canny_edges.png", edges);
    cout << "Canny边缘检测结果已保存为 canny_edges.png" << endl;

    // 显示结果 (可选)
    namedWindow("Canny Edges", WINDOW_NORMAL);
    imshow("Canny Edges", edges);
    waitKey(0);

    return 0;
}