#ifndef NNHDPROCESSOR_H
#define NNHDPROCESSOR_H

#include <opencv2/opencv.hpp>

class NNHDProcessor
{
public:
    NNHDProcessor();
    void main();
    void imgCompete();
    void imgUndoScale();

private:
    const double scaleFactor = 2.0;   //缩放参数
    const int patchSize = 5;
    cv::Mat imgOut;
    cv::Mat imgCompeted;
    cv::Mat imgWithoutScale;

};

#endif // NNHDPROCESSOR_H
