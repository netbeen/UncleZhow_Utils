#ifndef GENERATEHEIGHBORXML_H
#define GENERATEHEIGHBORXML_H

#include <opencv2/opencv.hpp>

class GenerateNeighborXML
{
public:
    GenerateNeighborXML();
    void main();

private:
    void generateRGBFeatures(const cv::Mat& sourceImage, cv::Mat& features , int patchSize);
    void generateHistogramFeatures(const cv::Mat& sourceImage, cv::Mat& features , int patchSize);
};

#endif // GENERATEHEIGHBORXML_H
