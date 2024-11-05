#include <opencv2/opencv.hpp>
#include <queue>
#include <iostream>

using namespace cv;
using namespace std;

// 定义一个结构体来存储坐标
struct Point2D {
    int x;
    int y;
    Point2D(int x, int y) : x(x), y(y) {}
};

// 区域生长算法
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
    queue<Point2D> pointQueue;
    pointQueue.push(Point2D(seed.x, seed.y));

    // 将种子点标记为已处理
    dst.at<uchar>(seed) = 255;  // 标记区域为白色

    // 开始区域生长
    while (!pointQueue.empty()) {
        Point2D currentPoint = pointQueue.front();
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
                    pointQueue.push(Point2D(newX, newY));

                    // 标记该点为区域的一部分
                    dst.at<uchar>(newY, newX) = 255;
                }
            }
        }
    }
}

int main() {
    // 读取灰度图像
    Mat src = imread("example.png", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "无法打开图像！" << endl;
        return -1;
    }

    // 初始化输出图像
    Mat dst;

    // 选择种子点和阈值
    Point seed(200, 200);  // 假设选择种子点为(100, 100)
    int threshold = 125;    // 灰度值阈值

    // 执行区域生长算法
    regionGrowing(src, dst, seed, threshold);

    // Step 3: 保存分割结果
    imwrite("./rg_result.png", dst);

    // 显示原图和分割结果
    imshow("Original Image", src);
    imshow("Segmented Image", dst);

    waitKey(0);
    return 0;
}