#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// MH边缘检测
int main() {
    // 读取图像
    Mat img = imread("./example.png", IMREAD_GRAYSCALE); 

    if (img.empty()) {
        cout << "无法读取图片！" << endl;
        return -1;
    }

    // Step 1: 高斯模糊 (平滑处理)
    Mat blurred;
    GaussianBlur(img, blurred, Size(5, 5), 1.0);

    // Step 2: 使用拉普拉斯算子 (Laplacian of Gaussian)
    Mat laplacian;
    Laplacian(blurred, laplacian, CV_16S, 3); // 拉普拉斯核大小为3
    Mat abs_laplacian;
    convertScaleAbs(laplacian, abs_laplacian); // 取绝对值

    // Step 3: 阈值处理找到边缘
    Mat edges;
    threshold(abs_laplacian, edges, 50, 255, THRESH_BINARY);

    // Step 4: 保存分割结果
    imwrite("./segmented_image.png", edges);
    cout << "分割后的图像已保存为 segmented_image.png" << endl;

    // 显示结果 (可选)
    namedWindow("Detected Edges", WINDOW_NORMAL);
    imshow("Detected Edges", edges);
    waitKey(0);

    return 0;
}