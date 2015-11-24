#include "findsimilarpoint.h"

FindSimilarPoint::FindSimilarPoint()
{

}

void FindSimilarPoint::GetFeatures(cv::Mat &features, cv::Mat &img, int patchSize)
{
    std::cout <<"img H: " << img.rows <<  "  img W: " << img.cols << std::endl;
    std::cout << "features H: " << features.rows << "features W: " << features.cols << std::endl;

    for(int topleft_y = 0; topleft_y < img.rows; topleft_y++){
        for(int topleft_x = 0; topleft_x < img.cols; topleft_x++){
            int trueTopleft_x = topleft_x;
            int trueTopleft_y = topleft_y;

            if(trueTopleft_x + this->patchSize >= img.cols){
                trueTopleft_x = img.cols-1-this->patchSize;
            }
            if(trueTopleft_y + this->patchSize >= img.rows){
                trueTopleft_y = img.rows-1-this->patchSize;
            }
            for(int offset_y = 0; offset_y < patchSize; offset_y++){
                for(int offset_x = 0; offset_x < patchSize; offset_x++){
                    features.at<float>((topleft_y*img.cols+topleft_x), (offset_y*patchSize+offset_x)*3+ 0) = (float)img.at<cv::Vec3b>(trueTopleft_y+offset_y,trueTopleft_x+offset_x)[0]/255.0/this->patchSize;
                    features.at<float>((topleft_y*img.cols+topleft_x) ,(offset_y*patchSize+offset_x)*3+ 1) = (float)img.at<cv::Vec3b>(trueTopleft_y+offset_y,trueTopleft_x+offset_x)[1]/255.0/this->patchSize;
                    features.at<float>((topleft_y*img.cols+topleft_x) ,(offset_y*patchSize+offset_x)*3+ 2) = (float)img.at<cv::Vec3b>(trueTopleft_y+offset_y,trueTopleft_x+offset_x)[2]/255.0/this->patchSize;
                }
            }
        }
    }
    //std::cout << features << std::endl;
}


void FindSimilarPoint::calcFeatureForSinglePoint(const cv::Mat& source, const cv::Point2i pointCoordinate, cv::Mat& feature){
   for(int y_offset = 0; y_offset < this->patchSize; y_offset++){
       for(int x_offset = 0; x_offset < this->patchSize; x_offset++){
            feature.at<float>(0,(y_offset*this->patchSize + x_offset) * 3 + 0) = (float)source.at<cv::Vec3b>( pointCoordinate.y + y_offset, pointCoordinate.x + x_offset)[0]/255.0 / patchSize;
            feature.at<float>(0,(y_offset*this->patchSize + x_offset) * 3 + 1) = (float)source.at<cv::Vec3b>( pointCoordinate.y + y_offset, pointCoordinate.x + x_offset)[1]/255.0 / patchSize;
            feature.at<float>(0,(y_offset*this->patchSize + x_offset) * 3 + 2) = (float)source.at<cv::Vec3b>( pointCoordinate.y + y_offset, pointCoordinate.x + x_offset)[2]/255.0 / patchSize;
        }
    }
}

void FindSimilarPoint::main(){
    cv::Mat img = cv::imread("/home/netbeen/桌面/img4.jpg");
    cv::resize(img,img,cv::Size(img.cols/this->scaleFactor, img.rows/this->scaleFactor));

    if(!img.isContinuous()) {
        std::cout<<"Error: not a continuous image!!!"<<std::endl;
    }

    int imgW = img.cols ;
    int imgH = img.rows;
    int numPts = imgW*imgH;
    int nChannels = img.channels();
    this->DIMENSION = 3*patchSize*patchSize;

    cv::Mat features = cv::Mat(numPts, DIMENSION, CV_32F);
    if(!features.isContinuous()) {
        std::cout<<"Error: not a continuous image!!!"<<std::endl;
    }

    this->GetFeatures(features, img, patchSize);
    //std::cout << features.row(1) << std::endl;

    // 3. 构建KDTree，计算search radius
    cv::flann::Index myKdTree;
    myKdTree.build(features, cv::flann::KDTreeIndexParams(16));

    cv::Mat imgDrawed = img.clone();
    imgDrawed.zeros(imgDrawed.size(),imgDrawed.type());

    cv::Point2i user_labeled_point = cv::Point2i(956/this->scaleFactor,630/this->scaleFactor);
    cv::Rect patchRect = cv::Rect(user_labeled_point.x,user_labeled_point.y,this->patchSize,this->patchSize);
    cv::Mat zoomInPatch = img(patchRect);
    cv::resize(zoomInPatch,zoomInPatch,cv::Size(128,128));
    cv::imshow("zoomInPatch",zoomInPatch);
    cv::imwrite("zoomInPatch.png",zoomInPatch);

    cv::Mat queryFeature = cv::Mat(1,this->DIMENSION,CV_32F);
    this->calcFeatureForSinglePoint(img,user_labeled_point,queryFeature);
    //std::cout << queryFeature << std::endl;

    //std::cout <<  "r: "<< this->GetSearchRadius(myKdTree,queryFeature,features.rows,0.02) << std::endl;
    //return;

    cv::Mat output;
    cv::Mat distance;
    //myKdTree.radiusSearch(queryFeature,output,distance,0.05847,features.rows,cv::flann::SearchParams(features.rows));
    myKdTree.knnSearch(queryFeature,output,distance,500,cv::flann::SearchParams(features.rows));
    //myKdTree.radiusSearch(queryFeature,output,distance,1,features.rows,cv::flann::SearchParams(features.rows));

    for(int i = 0; i < output.cols; i++){
        int index = output.at<int>(0,i);
        if(index == 0){
            break;
        }
        cv::circle(imgDrawed,cv::Point2i(index%imgDrawed.cols,index/imgDrawed.cols),1,cv::Scalar(255,0,0),2);
    }
    //cv::circle(imgDrawed,user_labeled_point,5,cv::Scalar(0,0,255),10);
    cv::rectangle(img,cv::Rect(user_labeled_point.x, user_labeled_point.y, this->patchSize, this->patchSize),cv::Scalar(0,0,255));

    cv::imshow("imgDrawed",imgDrawed);
    cv::imwrite("imgDrawed.png",imgDrawed);
    cv::imshow("img",img);
    cv::imwrite("img.png",img);

    cv::waitKey();

}

double FindSimilarPoint::GetSearchRadius(cv::flann::Index &myKdTree, cv::Mat &features, int nMaxSearch, float percent)
{
    std::cout << "FindSimilarPoint::GetSearchRadius start" << std::endl;
    //3.5 计算radius,目标2%
    double maxDistanceSum = 0.0;
    int numpts = features.rows;
    int DIMENSION = features.cols;
    int knn = numpts*percent;
    if(knn > nMaxSearch)
        knn = nMaxSearch;

    myKdTree.save("kd.kdkd");

    int nchecks = 2*knn > nMaxSearch ? 2*knn : nMaxSearch;
    for(int i = 0; i <  numpts; i++){
        std::cout << i << std::endl;
        std::cout << "now feature: " << features.row(i) << std::endl;
        cv::Mat indices;
        cv::Mat dists;
        myKdTree.knnSearch(features.row(i), indices, dists, knn, cv::flann::SearchParams(128));
        //std::cout << distances << std::endl;
        std::cout << "dists's size H W = " << dists.rows << " " << dists.cols << std::endl;
        float localMax = dists.at<float>(0, knn-1);

        //std::cout << localMax << std::endl;
        maxDistanceSum += localMax;
    }
    double search_radius = maxDistanceSum / numpts; //average
    std::cout << "search_radius = " << search_radius << std::endl;
    return search_radius;
}
