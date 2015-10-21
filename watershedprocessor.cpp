#include "watershedprocessor.h"

WatershedProcessor::WatershedProcessor()
{

}

void WatershedProcessor::watershed(const cv::Mat& inputImage, cv::Mat& inputOutputMark){
    cv::watershed(inputImage,inputOutputMark);
}


void WatershedProcessor::main(){
    cv::Mat img = cv::imread("/home/netbeen/桌面/img6.jpg");
    cv::Mat mark = cv::imread("/home/netbeen/桌面/mark6.png",cv::IMREAD_GRAYSCALE);
    mark.convertTo(mark,CV_32S);
    this->watershed(img,mark);
    mark.convertTo(mark,CV_8U);
    cv::imshow("mark",mark);
    cv::imwrite("mark.png",mark);
    cv::waitKey();
}
