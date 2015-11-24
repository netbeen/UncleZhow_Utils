#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

class Utils
{
public:
    Utils();

    static void doDensityPeakAndShow(const cv::Mat& features, const cv::Mat& rawImage, int maxClusters);
    static double GetSearchRadius(cv::flann::Index &myKdTree, const cv::Mat &features, int nMaxSearch, float percent);
    static int cmp(const std::pair<int, double> &x, const std::pair<int, double> &y );

};

#endif // UTILS_H
