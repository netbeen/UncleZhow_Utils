#ifndef WATERSHEDPROCESSOR_H
#define WATERSHEDPROCESSOR_H

#include <opencv2/opencv.hpp>

class WatershedProcessor
{
public:
    WatershedProcessor();
    void watershed(const cv::Mat& inputImage, cv::Mat& inputOutputMark);
    void main();

};

#endif // WATERSHEDPROCESSOR_H
