#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // 读取输入图像
    Mat image = imread("example.png"); 
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // 转换为灰度图像
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 高斯模糊去噪
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 自适应阈值获取二值图像
    Mat binary;
    threshold(blurred, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 形态学开运算去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat opening;
    morphologyEx(binary, opening, MORPH_OPEN, kernel, Point(-1, -1), 2);

    // 距离变换并阈值化获取前景区域
    Mat distTransform;
    distanceTransform(opening, distTransform, DIST_L2, 5);
    normalize(distTransform, distTransform, 0, 1.0, NORM_MINMAX); 
    threshold(distTransform, distTransform, 0.7, 1.0, THRESH_BINARY);
    Mat sureFg;
    distTransform.convertTo(sureFg, CV_8U, 255);

    // 膨胀以获取背景区域
    Mat sureBg;
    dilate(opening, sureBg, kernel, Point(-1, -1), 3);

    // 计算未知区域（背景减去前景）
    Mat unknown;
    subtract(sureBg, sureFg, unknown);

    // 连通组件标记
    Mat markers;
    connectedComponents(sureFg, markers);

    // 将所有标记加1，以确保背景标记为1
    markers = markers + 1;

    // 将未知区域标记为0
    for (int i = 0; i < unknown.rows; i++) {
        for (int j = 0; j < unknown.cols; j++) {
            if (unknown.at<uchar>(i, j) == 255) {
                markers.at<int>(i, j) = 0;
            }
        }
    }

    // 应用分水岭算法
    watershed(image, markers);

    // 将分割结果转换为二值图像
    Mat binaryResult = Mat::zeros(markers.size(), CV_8U);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) > 1) {
                binaryResult.at<uchar>(i, j) = 255;  // 将非背景区域设为白色
            }
        }
    }

    // 显示灰度图像和二值分割结果
    imshow("Grayscale Image", gray);
    imshow("Binary Segmentation Result (Watershed)", binaryResult);
    waitKey(0);

    return 0;
}