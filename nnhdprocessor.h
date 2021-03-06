#ifndef NNHDPROCESSOR_H
#define NNHDPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <QString>

class NNHDProcessor
{
public:
    NNHDProcessor();
    void main();
    void imgCompete();
    void imgUndoScale();

    void generateFeatureFromFile(const QString filename, cv::Mat& features);

private:
    const double scaleFactor = 2.0;   //缩放参数
    const int patchSize = 1;
    cv::Mat imgOut;
    cv::Mat imgCompeted;
    cv::Mat imgWithoutScale;

};

#endif // NNHDPROCESSOR_H
