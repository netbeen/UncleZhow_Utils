#ifndef PATCHMATCHPROCESSOR_H
#define PATCHMATCHPROCESSOR_H

#include <opencv2/opencv.hpp>

class PatchMatchProcessor
{
public:
    PatchMatchProcessor();
    void main();
    const int patchSize = 20;
    const double scaleFactor = 4.0;
    inline bool calcRotatedPatch(const cv::Mat& sourceImg, const cv::Point2i& topleft, const float angle, cv::Mat& targetPatch);
    inline bool ckeckValid(const cv::Mat& sourceImg, const cv::Mat2f& rotatedCoordinate);
    inline void bilinear(const cv::Mat& sourceImg, const cv::Mat2f& rotatedCoordinate, cv::Mat& targetPatch);
    inline  double calcL2Distance(const cv::Mat& patch1, const cv::Mat& patch2);
};

#endif // PATCHMATCHPROCESSOR_H
