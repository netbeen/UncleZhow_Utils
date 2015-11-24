#include "patchmatchprocessor.h"
#include <cmath>

PatchMatchProcessor::PatchMatchProcessor()
{

}

bool PatchMatchProcessor::ckeckValid(const cv::Mat& sourceImg, const cv::Mat2f& rotatedCoordinate){
    int width = sourceImg.cols;
    int height = sourceImg.rows;

    for(int offset_y = 0; offset_y < this->patchSize; offset_y++){
        for(int offset_x = 0; offset_x <  this->patchSize; offset_x++){
            if(rotatedCoordinate.at<cv::Vec2f>(offset_y, offset_x)[0] >= width || rotatedCoordinate.at<cv::Vec2f>(offset_y, offset_x)[0] < 0){
                return false;
            }
            if(rotatedCoordinate.at<cv::Vec2f>(offset_y, offset_x)[1] >= height || rotatedCoordinate.at<cv::Vec2f>(offset_y, offset_x)[1] < 0){
                return false;
            }
            //std::cout << offset_y << " " << offset_x << std::endl;
        }
    }
    return true;
}

//双线性差值
void PatchMatchProcessor::bilinear(const cv::Mat& sourceImg, const cv::Mat2f& rotatedCoordinate, cv::Mat& targetPatch){
    //std::cout << "bilinear start" << std::endl;
    assert(targetPatch.type() == CV_8UC3);
    assert(rotatedCoordinate.size() == targetPatch.size());
    for(int offset_y = 0; offset_y < this->patchSize; offset_y++){
        for(int offset_x = 0; offset_x < this->patchSize; offset_x++){
            cv::Point2f currentPoint = cv::Point2f(rotatedCoordinate.at<cv::Vec2f>(offset_y,offset_x)[0],rotatedCoordinate.at<cv::Vec2f>(offset_y,offset_x)[1]);
            cv::Point2i upleft = cv::Point2i(currentPoint.x, currentPoint.y);
            cv::Point2i upright = cv::Point2i(currentPoint.x+1, currentPoint.y);
            cv::Point2i lowleft = cv::Point2i(currentPoint.x, currentPoint.y+1);
            cv::Point2i lowright = cv::Point2i(currentPoint.x+1, currentPoint.y+1);

            float upleftArea = (currentPoint.x-upleft.x)*(currentPoint.y-upleft.y);
            float uprightArea = (upright.x - currentPoint.x)*(currentPoint.y-upright.y);
            float lowleftArea = (currentPoint.x - lowleft.x) * (lowleft.y - currentPoint.y);
            float lowrightArea = (lowright.x - currentPoint.x) * (lowright.y - currentPoint.y);

            targetPatch.at<cv::Vec3b>(offset_y,offset_x) = lowrightArea/1.0*sourceImg.at<cv::Vec3b>(upleft.y,upleft.x)+
                    lowleftArea/1.0*sourceImg.at<cv::Vec3b>(upright.y,upright.x)+
                    uprightArea/1.0*sourceImg.at<cv::Vec3b>(lowleft.y,lowleft.x)+
                    upleftArea/1.0*sourceImg.at<cv::Vec3b>(lowright.y,lowright.x);
            //targetPatch.at<cv::Vec3b>(offset_y,offset_x)[0] = 255;
            //std::cout << offset_y << " " << offset_x << std::endl;
        }
    }
    //std::cout << "bilinear done" << std::endl;
}

// coordinate 格式：（x,y）
 bool PatchMatchProcessor::calcRotatedPatch(const cv::Mat& sourceImg, const cv::Point2i& topleft, const float angle, cv::Mat& targetPatch){
    cv::Mat currentCoordinate = cv::Mat(this->patchSize, this->patchSize, CV_32FC2);
    for(int offset_y = 0; offset_y < this->patchSize; offset_y++){
        for(int offset_x = 0; offset_x < this->patchSize; offset_x++){
            currentCoordinate.at<cv::Vec2f>(offset_y, offset_x)[0] = topleft.x + offset_x;
            currentCoordinate.at<cv::Vec2f>(offset_y, offset_x)[1] = topleft.y + offset_y;
        }
    }

    //cv::Mat clone = sourceImg.clone();
    //cv::circle(clone,cv::Point2i(currentCoordinate.at<cv::Vec2f>(0,0)[0], currentCoordinate.at<cv::Vec2f>(0,0)[1]),1,cv::Scalar(255,0,0),2);
    //cv::circle(clone,cv::Point2i(currentCoordinate.at<cv::Vec2f>(this->patchSize-1,this->patchSize-1)[0], currentCoordinate.at<cv::Vec2f>(this->patchSize-1,this->patchSize-1)[1]),1,cv::Scalar(255,0,0),2);


    cv::Mat rotatedCoordinate = cv::Mat(this->patchSize, this->patchSize, CV_32FC2);
    for(int offset_y = 0; offset_y < this->patchSize; offset_y++){
        for(int offset_x = 0; offset_x < this->patchSize; offset_x++){
            float originalX = currentCoordinate.at<cv::Vec2f>(offset_y, offset_x)[0];
            float originalY = currentCoordinate.at<cv::Vec2f>(offset_y, offset_x)[1];
            float deltaX = originalX - topleft.x;
            float deltaY = originalY - topleft.y;
            rotatedCoordinate.at<cv::Vec2f>(offset_y, offset_x)[0] = topleft.x + cos(angle)*deltaX + sin(angle)*deltaY;
            rotatedCoordinate.at<cv::Vec2f>(offset_y, offset_x)[1] =  topleft.y - sin(angle) * deltaX + cos(angle) * deltaY;
        }
    }

    //std::cout << angle << std::endl;
    //std::cout << rotatedCoordinate.at<cv::Vec2f>(0,0)[0] << " " <<  rotatedCoordinate.at<cv::Vec2f>(0,0)[1]<< std::endl;
    //std::cout << rotatedCoordinate.at<cv::Vec2f>(this->patchSize-1,this->patchSize-1)[0] << " " <<  rotatedCoordinate.at<cv::Vec2f>(this->patchSize-1,this->patchSize-1)[1]<< std::endl;
    //cv::circle(clone,cv::Point2i(rotatedCoordinate.at<cv::Vec2f>(0,0)[0], rotatedCoordinate.at<cv::Vec2f>(0,0)[1]),1,cv::Scalar(0,255,0),2);
    //cv::circle(clone,cv::Point2i(rotatedCoordinate.at<cv::Vec2f>(this->patchSize-1,this->patchSize-1)[0], rotatedCoordinate.at<cv::Vec2f>(this->patchSize-1,this->patchSize-1)[1]),1,cv::Scalar(0,255,0),2);
    //cv::imshow("clone",clone);

    if(this->ckeckValid(sourceImg,rotatedCoordinate) == false){
        //std::cout << "this->ckeckValid(sourceImg,rotatedCoordinate) == false" << std::endl;
        return false;
    }

    this->bilinear(sourceImg,rotatedCoordinate,targetPatch);

    //std::cout << "calcRotatedPatch done" << std::endl;
    return true;
 }

 double PatchMatchProcessor::calcL2Distance(const cv::Mat& patch1, const cv::Mat& patch2){
    assert(patch1.size() == patch2.size());
    float distance = 0;
    for(int offset_y = 0; offset_y < patch1.rows; offset_y++){
        for(int offset_x = 0; offset_x < patch1.cols; offset_x++){
            distance+=pow(patch1.at<cv::Vec3b>(offset_y,offset_x)[0] - patch2.at<cv::Vec3b>(offset_y,offset_x)[0],2);
            distance+=pow(patch1.at<cv::Vec3b>(offset_y,offset_x)[1] - patch2.at<cv::Vec3b>(offset_y,offset_x)[1],2);
            distance+=pow(patch1.at<cv::Vec3b>(offset_y,offset_x)[2] - patch2.at<cv::Vec3b>(offset_y,offset_x)[2],2);
        }
    }
    return distance;
 }

 int my_cmp(std::pair<int,double> p1,std::pair<int,double>  p2)
 {
    return p1.second < p2.second;
 }

void PatchMatchProcessor::main(){
    const double Pi = 3.141592654;
    cv::Mat img = cv::imread("/home/netbeen/桌面/img4.jpg");
    //cv::Mat img = cv::imread("/home/netbeen/桌面/img7.jpg");
    cv::resize(img,img,cv::Size(img.cols/this->scaleFactor,img.rows/this->scaleFactor));


    const cv::Point2i user_labeled_point = cv::Point2i(956/this->scaleFactor,630/this->scaleFactor);
    //const cv::Point2i user_labeled_point = cv::Point2i(90/this->scaleFactor,159/this->scaleFactor);
    const cv::Rect patchRect = cv::Rect(user_labeled_point.x,user_labeled_point.y,this->patchSize,this->patchSize);
    const cv::Mat standardPatch = img(patchRect);

    /*cv::Mat targetPatchtest = cv::Mat(this->patchSize,this->patchSize,CV_8UC3);
    this->calcRotatedPatch(img,user_labeled_point,Pi/16,targetPatchtest);
    double distance = calcL2Distance(standardPatch,targetPatchtest);
    cv::resize(targetPatchtest,targetPatchtest,cv::Size(128,128));
    cv::imshow("targetPatchtest",targetPatchtest);
    std::cout << "distance: " << distance <<std::endl;*/


    std::vector<std::pair<int,double>> index2Distance;
    for(int offset_y = 0; offset_y < img.rows; offset_y++){
        for(int offset_x = 0; offset_x < img.cols; offset_x++){
            double localMin = 9999999;
            cv::Mat targetPatch = cv::Mat(this->patchSize,this->patchSize,CV_8UC3);
            for(float angle = 0; angle<= Pi/2; angle=angle+Pi/8){
                this->calcRotatedPatch(img,cv::Point2d(offset_x,offset_y),angle,targetPatch);
                double distance = calcL2Distance(standardPatch,targetPatch);
                if(distance < localMin){
                    localMin = distance;
                }
            }
            std::pair<int,double> temp = std::pair<int,double>(offset_y* img.cols+offset_x,localMin);
            index2Distance.push_back(temp);
        }
    }
    std::sort(index2Distance.begin(), index2Distance.end(), my_cmp);


    for(int i = 0 ; i < 2000; i++){
        //std::cout << index2Distance.at(i).first << std::endl;
        cv::circle(img,cv::Point2i(index2Distance.at(i).first%img.cols,index2Distance.at(i).first/img.cols),1,cv::Scalar(255,0,0),2);
    }

    cv::rectangle(img,cv::Rect(user_labeled_point.x, user_labeled_point.y, this->patchSize, this->patchSize),cv::Scalar(0,0,255));
    cv::imshow("img",img);
    cv::imwrite("img.png",img);
    cv::waitKey();
}
