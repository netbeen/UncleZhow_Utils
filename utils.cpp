#include "utils.h"

using namespace std;

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

int Utils::cmp(const std::pair<int, double> &x, const std::pair<int, double> &y )
{
    return x.second > y.second;
}


Utils::Utils()
{

}

double Utils::GetSearchRadius(cv::flann::Index &myKdTree, const cv::Mat &features, int nMaxSearch, float percent = 0.02f)
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


void Utils::doDensityPeakAndShow(const cv::Mat& features, const cv::Mat& rawImage, int maxClusters){
    assert(features.rows == rawImage.rows*rawImage.cols);

    // 0. 全局参数
        int nMaxSearch = 128;

        int imgW = rawImage.cols ;
        int imgH = rawImage.rows;
        int numPts = imgW*imgH;
        //int nChannels = rawImage.channels();
        int DIMENSION = features.cols;

    // 3. 构建KDTree，计算search radius
    cv::flann::Index myKdTree;
    myKdTree.build(features, cv::flann::KDTreeIndexParams(nMaxSearch));

    double search_radius = Utils::GetSearchRadius(myKdTree, features, nMaxSearch, 0.10);
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
            cout<<"KdTree failed, current i =" << i <<endl;
            exit(0);
        }
        v_density_Descend[i] = std::pair<int, double> (i, (double) nofNeighbors-1.0);

        v_points[i].nID = i;
        v_points[i].density = v_density_Descend[i].second;
    }

    // 5. Sort density in descending order
    std::sort(v_density_Descend.begin(), v_density_Descend.end(), Utils::cmp);

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

        if(iter->label == 0) // global maximum density
            continue;

        std::vector<float> queryPos(DIMENSION, 0.0);
        for(int k=0; k<DIMENSION; k++)
            queryPos[k] = pf[i*DIMENSION+k];

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
    cv::Mat imgOut = cv::Mat(imgH, imgW, CV_8UC3);
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

    cv::imshow("Input", rawImage);
    cv::imshow("Output", imgOut);
}
