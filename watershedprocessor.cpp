#include "watershedprocessor.h"
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

WatershedProcessor::WatershedProcessor()
{

}

void WatershedProcessor::watershed(const cv::Mat& inputImage, cv::Mat& inputOutputMark){
    cv::watershed(inputImage,inputOutputMark);
}


void saveInVector(std::vector<std::pair<cv::Vec3b,int>>& neighborColor, cv::Vec3b inputColor){
    for(int i = 0; i < neighborColor.size(); i++){
        if(neighborColor.at(i).first == inputColor){
            neighborColor.at(i).second++;
            return;
        }
    }
    std::pair<cv::Vec3b,int> pair = std::pair<cv::Vec3b,int>(inputColor, 1);
    neighborColor.push_back(pair);
}

int my_cmp20151121(std::pair<cv::Vec3b,int> p1,std::pair<cv::Vec3b,int>  p2){
    return p1.second > p2.second;
}

//2015年11月21日的函数：给各个灰度mark上色，并且进行后期处理
void function20151121(const cv::Mat& markGray){
    cv::Mat markColor = cv::Mat(markGray.size(), CV_8UC3);
    for(int y_offset = 0; y_offset < markColor.rows; y_offset++){
        for(int x_offset = 0; x_offset < markColor.cols; x_offset++){
            if(markGray.at<uchar>(y_offset,x_offset) == 50){
                markColor.at<cv::Vec3b>(y_offset,x_offset)[0] = 255;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[1] = 0;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[2] = 255;
            }else if(markGray.at<uchar>(y_offset,x_offset) == 155){
                markColor.at<cv::Vec3b>(y_offset,x_offset)[0] = 0;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[1] = 255;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[2] = 255;
            }else if(markGray.at<uchar>(y_offset,x_offset) == 116){
                markColor.at<cv::Vec3b>(y_offset,x_offset)[0] = 0;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[1] = 255;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[2] = 0;
            }else if(markGray.at<uchar>(y_offset,x_offset) == 218){
                markColor.at<cv::Vec3b>(y_offset,x_offset)[0] = 255;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[1] = 0;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[2] = 0;
            }else if(markGray.at<uchar>(y_offset,x_offset) == 0){
                markColor.at<cv::Vec3b>(y_offset,x_offset)[0] = 0;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[1] = 0;
                markColor.at<cv::Vec3b>(y_offset,x_offset)[2] = 0;
            }else{
                std::cout << "current gray: " << (int)markGray.at<uchar>(y_offset,x_offset) << std::endl;
                std::cout << "ERROR" << std::endl;
            }
        }
    }


    cv::Mat markColorNoBlack = cv::Mat(markGray.size(), CV_8UC3);
    for(int y_offset = 0; y_offset < markColor.rows; y_offset++){
        for(int x_offset = 0; x_offset < markColor.cols; x_offset++){
            if(markColor.at<cv::Vec3b>(y_offset,x_offset)[0] == 0 && markColor.at<cv::Vec3b>(y_offset,x_offset)[1] == 0 &&markColor.at<cv::Vec3b>(y_offset,x_offset)[2] == 0){
                int x_offset_true = x_offset;
                int y_offset_true = y_offset;
                if(x_offset_true == markColor.cols/2 && y_offset_true == markColor.rows/2){
                    std::cout << "CENTER BLACK" << std::endl;
                }
                while(markColor.at<cv::Vec3b>(y_offset_true,x_offset_true)[0] == 0 &&
                      markColor.at<cv::Vec3b>(y_offset_true,x_offset_true)[1] == 0 &&
                      markColor.at<cv::Vec3b>(y_offset_true,x_offset_true)[2] == 0  &&
                      (x_offset_true != markColor.cols/2 || y_offset_true != markColor.rows/2)){
                    if(x_offset_true > markColor.cols/2){
                        x_offset_true--;
                    }else{
                        x_offset_true++;
                    }
                    if(y_offset_true > markColor.rows/2){
                        y_offset_true--;
                    }else{
                        y_offset_true++;
                    }
                }
                markColorNoBlack.at<cv::Vec3b>(y_offset,x_offset) = markColor.at<cv::Vec3b>(y_offset_true,x_offset_true);
            }else{
                markColorNoBlack.at<cv::Vec3b>(y_offset,x_offset) = markColor.at<cv::Vec3b>(y_offset,x_offset);
            }
        }
    }


    cv::Mat markColorNoBlackSmooth = markColorNoBlack.clone();
    for(int y_offset = 1; y_offset < markColor.rows-1; y_offset++){
        for(int x_offset = 1; x_offset < markColor.cols-1; x_offset++){
            std::vector<std::pair<cv::Vec3b,int>> neighborColor;
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset-1,x_offset-1));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset-1,x_offset-0));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset-1,x_offset+1));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset-0,x_offset-1));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset-0,x_offset+1));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset+1,x_offset-1));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset+1,x_offset-0));
            saveInVector(neighborColor,markColorNoBlack.at<cv::Vec3b>(y_offset+1,x_offset+1));

            std::sort(neighborColor.begin(), neighborColor.end(), my_cmp20151121);
            std::cout << neighborColor.size() << neighborColor.front().second << neighborColor.back().second << std::endl;

            markColorNoBlackSmooth.at<cv::Vec3b>(y_offset,x_offset) = neighborColor.front().first;
        }
    }

    cv::imshow("markColor",markColor);
    cv::imwrite("markColor.png",markColor);

    cv::imshow("markColorNoBlack",markColorNoBlack);
    cv::imwrite("markColorNoBlack.png",markColorNoBlack);

    cv::imshow("markColorNoBlackSmooth",markColorNoBlackSmooth);
    cv::imwrite("markColorNoBlackSmooth.png",markColorNoBlackSmooth);
}

void WatershedProcessor::main(){
    cv::Mat img = cv::imread("/home/netbeen/桌面/周叔项目/color histogram（256bins），textons（200  to 200 bins）(patchSize=16).png");
    cv::Mat mark = cv::imread("/home/netbeen/桌面/周叔项目/color histogram（256bins），textons（200  to 200 bins）(patchSize=16)-mark.png",cv::IMREAD_GRAYSCALE);
    mark.convertTo(mark,CV_32S);
    this->watershed(img,mark);
    mark.convertTo(mark,CV_8U);
    cv::imshow("mark",mark);
    cv::imwrite("mark.png",mark);
    //std::cout << mark.row(1) << std::endl;

    //std::cout << "0 0 gray: " << static_cast<int>(mark.at<uchar>(0,0)) << std::endl;
    function20151121(mark);

    cv::waitKey();
}
