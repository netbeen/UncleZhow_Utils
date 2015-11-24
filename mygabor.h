#ifndef MYGABOR_H
#define MYGABOR_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>



using namespace cv;
using namespace std;

class MyGabor
{
public:
    MyGabor();
    void main();
    const double scaleFactor = 2.0;   //缩放参数

    const double PI = 3.14159265;
    Mat getMyGabor(int width, int height, int U, int V, double Kmax, double f,double sigma, int ktype, const string kernel_name);
    void construct_gabor_bank();
    Mat gabor_filter(Mat& img, int type);
    cv::Mat generateFeature(const cv::Mat& gaborImg);
    static int cmp(const std::pair<int, double> &x, const std::pair<int, double> &y );

};

#endif // MYGABOR_H
