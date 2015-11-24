#include "shj_kdtree.h"
#include <fstream>

SHJ_KDTree::SHJ_KDTree()
{

}

void SHJ_KDTree::main(){
    cv::Mat trainFeatures;
    cv::Mat testFeatures;
    this->loadFeatrueFromFile("/home/netbeen/桌面/Kcluster.txt",trainFeatures,200,8);
    this->loadFeatrueFromFile("/home/netbeen/桌面/samples.txt",testFeatures,262144,8);

    this->myKdTree.build(trainFeatures, cv::flann::KDTreeIndexParams(512));

    std::ofstream outputFile("output.txt",std::ios::out);
    for(int i = 0; i< testFeatures.rows; i++){
        cv::Mat output;
        cv::Mat distance;
        this->myKdTree.knnSearch(testFeatures.row(i),output,distance,1,cv::flann::SearchParams(512));
        outputFile << output.at<int>(0,0) << std::endl;
    }
    outputFile.close();


}

void SHJ_KDTree::loadFeatrueFromFile(QString filename, cv::Mat& features, int initRows, int initCols){
    features = cv::Mat(initRows,initCols,CV_32F);

    std::ifstream file(filename.toUtf8().constData());
    double tempD;
    for(int row_index = 0; row_index < initRows; row_index++){
        for(int col_index = 0; col_index < initCols; col_index++){
            file >> tempD;
            features.at<float>(row_index,col_index) = tempD;
        }
    }
    file.close();

}
