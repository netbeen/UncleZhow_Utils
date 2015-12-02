#ifndef READSUPERPIXELDAT_H
#define READSUPERPIXELDAT_H

#include <opencv2/opencv.hpp>


class ReadSuperPixelDat
{
public:
    ReadSuperPixelDat();
    void main();
    void authorSegment();
    void authorRead();

private:
    cv::Mat image;
    cv::Mat result;
    int size;
    unsigned int* pbuff;
    void ReadImage(unsigned int* pbuff, int width,int height);
    void SaveSegmentedImageFile(unsigned int* pbuff, int width,int height);
    int labels[INT32_MAX];
    void analyseLabelFile();
    void searchWriteArea();
    cv::Mat maskDFS;
    cv::Mat mask;

    void dfs(std::pair<int,int> coordinate);

    std::vector<std::vector< std::pair<int,int> > > maskID2Coodinates;
    std::vector< std::pair<int,int> > coordinates;
};

#endif // READSUPERPIXELDAT_H
