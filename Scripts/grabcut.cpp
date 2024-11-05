#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 读取图像
    Mat image = imread("./grab_example.png");
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // 定义一个矩形框选区域（用于初始前景）
    // 注意：这里的Rect参数需要根据实际图像内容调整
    Rect rectangle(50, 100, 350, 330);  

    // 创建掩码、前景模型和背景模型
    Mat mask, bgdModel, fgdModel;

    // 使用GrabCut算法分割图像
    grabCut(image, mask, rectangle, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);

    // 提取前景区域
    compare(mask, GC_PR_FGD, mask, CMP_EQ);
    Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
    image.copyTo(foreground, mask);

    // 显示结果
    imshow("Original Image", image);
    imshow("Binary Foreground (GrabCut Result)", foreground);
    waitKey(0);

    return 0;
}