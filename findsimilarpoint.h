#ifndef FINDSIMILARPOINT_H
#define FINDSIMILARPOINT_H

#include <opencv2/opencv.hpp>
#include <cstdlib>

class FindSimilarPoint
{
public:
    FindSimilarPoint();
    void main();

    void GetFeatures(cv::Mat &features, cv::Mat &img, int patchSize);
    //void get_rand(double *p, int n);//函数功能为产生n个0-1的随机数，存储于数组p中。
    void calcFeatureForSinglePoint(const cv::Mat& source, const cv::Point2i pointCoordinate, cv::Mat& feature);
    double GetSearchRadius(cv::flann::Index &myKdTree, cv::Mat &features, int nMaxSearch, float percent);

    const int patchSize = 20;
    const double scaleFactor = 4.0;
    int DIMENSION;
};

#endif // FINDSIMILARPOINT_H
