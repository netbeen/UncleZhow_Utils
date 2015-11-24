#ifndef SHJ_KDTREE_H
#define SHJ_KDTREE_H

#include <opencv2/opencv.hpp>
#include <QString>

class SHJ_KDTree
{
public:
    SHJ_KDTree();
    void main();
    void loadFeatrueFromFile(QString filename, cv::Mat& features, int initRows, int initCols);

private:
    cv::flann::Index myKdTree;
};

#endif // SHJ_KDTREE_H
