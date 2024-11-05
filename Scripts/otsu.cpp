// 大津法
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 读取图像（灰度模式）
    Mat image = imread("example.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "无法打开图像文件！" << endl;
        return -1;
    }

    // 创建保存大津法结果的矩阵
    Mat otsu_result;

    // 使用大津法进行图像二值化，返回阈值
    double otsu_thresh_val = threshold(image, otsu_result, 0, 255, THRESH_BINARY + THRESH_OTSU);

    // 输出自动计算的大津阈值
    cout << "Otsu's Threshold Value: " << otsu_thresh_val << endl;

    // 显示原始图像和分割结果
    imshow("Original Image", image);
    imshow("Otsu Threshold Image", otsu_result);

    // Step 3: 保存分割结果
    imwrite("./otsu_result.png", otsu_result);
    cout << "Otsu分割结果已保存为otsu_result.png" << endl;

    // 等待用户按键关闭窗口
    waitKey(0);
    return 0;
}