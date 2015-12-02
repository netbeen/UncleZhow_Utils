#include "nnhdprocessor.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
using namespace  std;

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

#include <cstdlib>
#include <ctime>
#include <time.h>

void get_rand(double *p, int n)//函数功能为产生n个0-1的随机数，存储于数组p中。
{
    int i;
#define N 9999 //三位小数。
    srand(time(NULL));//设置随机数种子，使每次获取的随机序列不同。
    for(i = 0; i < n; i ++)
        p[i] = (rand()%(N+1)/(double)(N+1)) / 100.0;//生成0-1间的随机数。
}

typedef struct  tagMyClusteringTestPoint
{
    int nID, nNNHD, label;
    double density; //rho
    double dis2NNHD; // delta

    tagMyClusteringTestPoint()
    {
        nID = nNNHD = label = -1;
        density = 0.0;
        dis2NNHD = 999999.0;
    }
} ClusteringPoints;

int cmp(const std::pair<int, double> &x, const std::pair<int, double> &y )
{
    return x.second > y.second;
}

void GetFeatures(cv::Mat &features, cv::Mat &img, int patchSize)
{
    std::cout << "Start GetFeatures" << std::endl;
    int imgW = img.cols - patchSize + 1;
    int imgH = img.rows - patchSize + 1;
    int numPts = imgW*imgH;
    int nChannels = img.channels();
    int DIMENSION = nChannels*patchSize*patchSize;

    features = cv::Mat(numPts, DIMENSION, CV_32F);

    for(int y_offset = 0; y_offset < imgH; y_offset++)
        for(int x_offset = 0; x_offset < imgW; x_offset++)
            for(int y_offset_patch = 0; y_offset_patch < patchSize; y_offset_patch++)
                for(int x_offset_patch = 0; x_offset_patch < patchSize; x_offset_patch++){
                    cv::Vec3b currentColor = img.at<cv::Vec3b>(y_offset+y_offset_patch,x_offset+x_offset_patch);
                    features.at<float>(y_offset*imgW+x_offset, y_offset_patch*patchSize+x_offset_patch+0) = (float)currentColor[0];
                    features.at<float>(y_offset*imgW+x_offset, y_offset_patch*patchSize+x_offset_patch+1) = (float)currentColor[1];
                    features.at<float>(y_offset*imgW+x_offset, y_offset_patch*patchSize+x_offset_patch+2) = (float)currentColor[2];
                }
}

double GetSearchRadius(cv::flann::Index &myKdTree, cv::Mat &features, int nMaxSearch, float percent = 0.02f)
{
    //3.5 计算radius,目标2%
    double maxDistanceSum = 0.0;
    int numpts = features.rows;
    int DIMENSION = features.cols;
    int knn = numpts*percent;
    if(knn > nMaxSearch)
        knn = nMaxSearch;

    int nchecks = 2*knn > nMaxSearch ? 2*knn : nMaxSearch;
    for(int i = 0; i <  numpts; i++){
        cv::Mat indices;
        cv::Mat dists;
        myKdTree.knnSearch(features.row(i), indices, dists, knn, cv::flann::SearchParams(nchecks));
        //std::cout << distances << std::endl;
        float localMax = dists.at<float>(0, knn-1);

        //std::cout << localMax << std::endl;
        maxDistanceSum += localMax;
    }
    double search_radius = maxDistanceSum / numpts; //average
    return search_radius;
}

NNHDProcessor::NNHDProcessor()
{

}

void NNHDProcessor::generateFeatureFromFile(const QString filename, cv::Mat& features){
    std::cout << "WARNING: load featrue from file! Cannot display picture!" << std::endl;
    ifstream file(filename.toUtf8().constData());

    const int ROWS = 45;
    const int COLS = 701;

    features = cv::Mat(ROWS,COLS,CV_32F);
    double tempD;
    for(int row_index = 0; row_index < ROWS; row_index++){
        for(int col_index = 0; col_index < COLS; col_index++){
            file >> tempD;
            features.at<float>(row_index,col_index) = tempD;
        }
    }
}

void NNHDProcessor::main(){
    // 0. 全局参数
        int maxClusters = 2;
        double search_radius =  0.08; //; + 0.01*(iteration+1);
        int nMaxSearch = 128;


        // 1. 读入影像
        string imgfile = "/home/netbeen/桌面/周叔项目/img3.jpg";
        cv::Mat img = cv::imread(imgfile);
        int imgRescaleW = img.cols/this->scaleFactor;
        int imgRescaleH = img.rows/this->scaleFactor;
        cv::resize(img, img, cv::Size(imgRescaleW, imgRescaleH));


        clock_t start_time=clock();

        // 2. 取出所有点的rgb构成特征空间
        if(!img.isContinuous()) {
            cout<<"Error: not a continuous image!!!"<<endl;
        }

        int imgW = img.cols - patchSize + 1;
        int imgH = img.rows - patchSize + 1;
        int numPts = imgW*imgH;
        int nChannels = img.channels();
        int DIMENSION = patchSize*patchSize*nChannels;

        cv::Mat features;
        GetFeatures(features, img, patchSize);

        // 3. 构建KDTree，计算search radius
        cv::flann::Index myKdTree;
        myKdTree.build(features, cv::flann::KDTreeIndexParams(128));

        search_radius = GetSearchRadius(myKdTree, features, nMaxSearch);
        std::cout << " Search Radius: " << search_radius << std::endl;

        // 4. 计算Density - radius search
        float* pf = (float*) features.data;

        std::vector<ClusteringPoints> v_points;
        ClusteringPoints pt;
        v_points.assign(numPts, pt);

        std::vector<std::pair<int, double> > v_density_Descend;
        v_density_Descend.assign(numPts, std::pair<int, double>(-1, 0.0));

        for(int i=0; i<numPts; i++) {
            std::vector<float> queryPos(DIMENSION, 0.0);
            for(int k=0; k<DIMENSION; k++)
                queryPos[k] = pf[i*DIMENSION+k];

            std::vector<int> indices;
            std::vector<float> dists;
            int nofNeighbors = myKdTree.radiusSearch(queryPos, indices, dists, search_radius, numPts, cv::flann::SearchParams(nMaxSearch));
            if(nofNeighbors < 1) {
                cout<<"KdTree failed"<<endl;
                exit(0);
            }
            v_density_Descend[i] = std::pair<int, double> (i, (double) nofNeighbors-1.0);

            v_points[i].nID = i;
            v_points[i].density = v_density_Descend[i].second;
        }

        // 5. Sort density in descending order
        std::sort(v_density_Descend.begin(), v_density_Descend.end(), cmp);

        std::vector<int> rankinDesityDescend; // record the rank of the i-th point in the descending order density list
        rankinDesityDescend.assign(numPts, -1);
        for(int i=0; i<numPts; i++) {
            int index = v_density_Descend[i].first;
            rankinDesityDescend[index] = i;
        }

        int nglobalMax = v_density_Descend.begin()->first;
        v_points[nglobalMax].label = 0;

        // 6. 计算 dis2NNHD
        std::vector<ClusteringPoints>::iterator iter = v_points.begin();
        for(int i=0; i<numPts; i++, iter++) {
            //cout<<i<<endl;

            if(iter->label == 0) // global maximum density
                continue;

            std::vector<float> queryPos(DIMENSION, 0.0);
            for(int k=0; k<DIMENSION; k++)
                queryPos[k] = pf[i*DIMENSION+k];
            //cout<<"queryPos = "<<queryPos<<endl;

            int knn = 16;
            int startindex = 1;
            int curRank = rankinDesityDescend[i];

            while (1) {
                if (curRank < 2*knn || knn >= 512) { // no need to search in knn neighborhood
                    double minDis = 999999.0;
                    int nNNHD = -1;
                    for(int j=0; j<curRank; j++) {
                        int index = v_density_Descend[j].first;
                        double dis = 0.0;
                        for(int k=0; k<DIMENSION; k++) {
                            double d = pf[index*DIMENSION+k] - queryPos[k];
                            dis +=  d*d;
                        }
                        if(dis < minDis) {
                            minDis = dis;
                            nNNHD = index;
                        }
                    }
                    iter->nNNHD = nNNHD;
                    iter->dis2NNHD = minDis;

                    break;
                }
                else {
                    std::vector<int> indices;
                    std::vector<float> dists;
                    int nchecks = nMaxSearch;
                    if(nchecks < 2*knn)
                        nchecks = 2*knn;
                    myKdTree.knnSearch(queryPos, indices, dists, knn, cv::flann::SearchParams(nchecks)); //, cv::flann::SearchParams(knn+1));
                    bool bfound = false;
                    double distest1 = dists[0];
                    for(int j=startindex; j<knn; j++) {
                        int nnIndex = indices[j];
                        if(nnIndex < 0 || nnIndex >= numPts) {
                            cout<<"knn search error"<<endl;
                            exit(0);
                        }

                        double distest2 = dists[j];
                        double dis = 0.0;

                        if(rankinDesityDescend[nnIndex] < curRank) { //have found the nearest neighbor of higher density
                            iter->nNNHD = nnIndex;
                            iter->dis2NNHD = distest2;
                            bfound = true;
                            break;
                        }
                    }
                    if(bfound)
                        break;
                    else {
                        startindex = knn;
                        knn = knn*2 < numPts ? knn*2 : numPts;
                    }
                }
            }
        }

        // 7. 计算 delta*rho = dis2nnhd*density
        std::vector<std::pair<int, double> > v_deltarho;
        v_deltarho.assign(numPts, std::pair<int, double>(-1, 0.0));
        std::vector<std::pair<int, double> >::iterator iter_deltaRho = v_deltarho.begin();
        iter = v_points.begin();
        double maxDelta = 0;
        for(int i=0; i<numPts; i++, iter_deltaRho++, iter++) {
            iter_deltaRho->first = i;
            iter_deltaRho->second = iter->density*iter->dis2NNHD;
            if(iter->dis2NNHD > maxDelta && iter->label != 0)
                maxDelta = iter->dis2NNHD;
        }
        v_points[nglobalMax].dis2NNHD = maxDelta + search_radius;

        // sort delta*rho
        sort(v_deltarho.begin(), v_deltarho.end(), cmp);

        std::vector<int> centerofCluster;
        centerofCluster.push_back(v_density_Descend.begin()->first);

        int nClusters = 1;
        for(int i=1; i<maxClusters; i++) {
            int curID = v_deltarho[i].first;
            v_points[curID].label = nClusters;
            centerofCluster.push_back(curID);
            nClusters++;

            if(v_deltarho[i].second  > 2.0*v_deltarho[i+1].second)
                break;
        }

        cout<<"num of clusters: "<<nClusters<<endl;

        // assign labels for other points
        // query points through the density list of descending order
        std::vector<std::pair<int, double> >::iterator iter_d_descend = v_density_Descend.begin();
        iter_d_descend = v_density_Descend.begin();
        for(int i=0; i<numPts; i++, iter_d_descend++) {
            int curID = iter_d_descend->first;
            if(v_points[curID].label != -1)
                continue;

            int nNNHD = v_points[curID].nNNHD;
            v_points[curID].label = v_points[nNNHD].label;
        }

        // output
        imgOut = cv::Mat(imgH, imgW, CV_8UC3);
        if(!imgOut.isContinuous()) {
            cout<<"not continous"<<endl;
            exit(0);
        }

        std::vector<int> icounts(nClusters, 0);
        double step = 255.0 / nClusters;
        uchar* outdata = (uchar*)imgOut.data;
        iter = v_points.begin();
        for(int i=0; i<imgH; i++) {
            for(int j=0; j<imgW; j++, iter++) {
                int label = iter->label;
                if(label == -1) {
                    cout<<"label == -1 at "<<i*imgW+j<<endl;
                    exit(0);
                }

                icounts[label] += 1;

                double r = step*(nClusters - label);
                double g = step*label;
                double b = 128*(label % 2);

                outdata[(i*imgW+j)*3 + 0] = (uchar) r;
                outdata[(i*imgW+j)*3 + 1] = (uchar) g;
                outdata[(i*imgW+j)*3 + 2] = (uchar) 0;
            }
        }

        for(int i=0; i<nClusters; i++)
            cout<<"cluster center "<<centerofCluster[i]<<endl;
        for(int i=0; i<nClusters; i++)
            cout<<"pixel num: "<<icounts[i]<<endl;

        clock_t end_time=clock();
        cout<< "Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时

        cv::imshow("Input", img);
        cv::imshow("Output", imgOut);



        this->imgCompete();
        //this->imgUndoScale();


        cv::imwrite("Input.png", img);
        cv::imwrite("imgCompeted.png", this->imgCompeted);

        waitKey();
}


void NNHDProcessor::imgCompete(){
    this->imgCompeted = cv::Mat(this->imgOut.rows + this->patchSize, this->imgOut.cols + this->patchSize,CV_8UC3);
    for(int i = 0; i < this->imgOut.rows; i++){
        for(int j = 0; j < this->imgOut.cols; j++){
            this->imgCompeted.at<cv::Vec3b>(i,j)[0] = this->imgOut.at<cv::Vec3b>(i,j)[0];
            this->imgCompeted.at<cv::Vec3b>(i,j)[1] = this->imgOut.at<cv::Vec3b>(i,j)[1];
            this->imgCompeted.at<cv::Vec3b>(i,j)[2] = this->imgOut.at<cv::Vec3b>(i,j)[2];
        }
    }
    for(int i = 0; i < this->patchSize; i++){
        for(int j = 0; j < this->imgOut.rows; j++){
            this->imgCompeted.at<cv::Vec3b>(j,imgOut.cols+i)[0] = this->imgOut.at<cv::Vec3b>(j,imgOut.cols-1)[0];
            this->imgCompeted.at<cv::Vec3b>(j,imgOut.cols+i)[1] = this->imgOut.at<cv::Vec3b>(j,imgOut.cols-1)[1];
            this->imgCompeted.at<cv::Vec3b>(j,imgOut.cols+i)[2] = this->imgOut.at<cv::Vec3b>(j,imgOut.cols-1)[2];
        }
    }
    for(int i = 0; i < this->patchSize; i++){
        for(int j = 0; j < this->imgCompeted.cols; j++){
            this->imgCompeted.at<cv::Vec3b>(this->imgOut.rows+i, j)[0] = this->imgCompeted.at<cv::Vec3b>(this->imgOut.rows-1,j)[0];
            this->imgCompeted.at<cv::Vec3b>(this->imgOut.rows+i, j)[1] = this->imgCompeted.at<cv::Vec3b>(this->imgOut.rows-1,j)[1];
            this->imgCompeted.at<cv::Vec3b>(this->imgOut.rows+i, j)[2] = this->imgCompeted.at<cv::Vec3b>(this->imgOut.rows-1,j)[2];
        }
    }
    cv::imshow("imgCompeted",this->imgCompeted);
}

void NNHDProcessor::imgUndoScale(){
    this->imgWithoutScale = cv::Mat(imgCompeted.rows*this->scaleFactor,imgCompeted.cols*this->scaleFactor,imgCompeted.type());
    for(int i = 0; i < this->imgCompeted.rows; i++){
        for(int j = 0; j < this->imgCompeted.cols; j++){
            for(int patchX = 0; patchX < this->patchSize; patchX++){
                for(int patchY = 0; patchY < this->patchSize; patchY++){
                    if(((i*this->scaleFactor + patchX) >= this->imgWithoutScale.rows) || ((j *this->scaleFactor +patchY) >= this->imgWithoutScale.cols)){
                        continue;
                    }
                    this->imgWithoutScale.at<cv::Vec3b>(i*this->scaleFactor + patchX, j *this->scaleFactor +patchY)[0] = this->imgCompeted.at<cv::Vec3b>(i,j)[0];
                    this->imgWithoutScale.at<cv::Vec3b>(i*this->scaleFactor + patchX, j *this->scaleFactor +patchY)[1] = this->imgCompeted.at<cv::Vec3b>(i,j)[1];
                    this->imgWithoutScale.at<cv::Vec3b>(i*this->scaleFactor + patchX, j *this->scaleFactor +patchY)[2] = this->imgCompeted.at<cv::Vec3b>(i,j)[2];
                }
            }
        }
    }
    cv::imshow("imgWithoutScale",imgWithoutScale);
    cv::imwrite("label.png",imgWithoutScale);
}
