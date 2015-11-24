#include "thresholdprocessor.h"

ThresholdProcessor::ThresholdProcessor()
{

}

void ThresholdProcessor::main(){
    cv::Mat img = cv::imread("/home/netbeen/桌面/img7.jpg",cv::IMREAD_GRAYSCALE);
    cv::threshold(img,img,0,255,cv::THRESH_OTSU);
    cv::dilate(img,img,cv::Mat());
    cv::erode(img,img,cv::Mat());
    cv::erode(img,img,cv::Mat());
    cv::dilate(img,img,cv::Mat());
    cv::dilate(img,img,cv::Mat());
    cv::erode(img,img,cv::Mat());
    //cv::dilate(img,img,cv::Mat());
    cv::imshow("img",img);
    cv::imwrite("binary.png",img);
    cv::waitKey();
}
