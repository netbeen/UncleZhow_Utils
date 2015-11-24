#ifndef GRAPHCUT_H
#define GRAPHCUT_H

#include <opencv2/opencv.hpp>


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "gco-v3.0/GCoptimization.h"

class GraphCut
{
public:
    GraphCut();
    void main();
    void main2();

private:
    const int CLASS_NUMBER = 2;
    const int scaleFactor = 1;
    cv::Mat initGuessGray;
    cv::Mat rawImage;
    cv::Mat resultLabelGray;
    std::vector<uchar> label2GrayValue;

    bool checkUserMarkValid(const cv::Mat& userMark);
    void GridGraph_Individually(int width,int height,int num_pixels,int num_labels);
};

#endif // GRAPHCUT_H
