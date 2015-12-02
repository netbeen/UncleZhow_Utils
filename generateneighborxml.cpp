#include "generateneighborxml.h"
#include <QDomDocument>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <climits>
#include <fstream>

GenerateNeighborXML::GenerateNeighborXML()
{

}

void GenerateNeighborXML::main(){
    const cv::Mat rawImage = cv::imread("/home/netbeen/桌面/周叔项目/exemplar.png");
    std::cout << "rawImage width: " << rawImage.cols << ", height: " << rawImage.rows << std::endl;

    const int patchSize = 10;

    const int normalizeWidth = rawImage.cols/patchSize*patchSize;
    const int normalizeHeight= rawImage.rows/patchSize*patchSize;
    cv::Mat normalizeImage = cv::Mat(normalizeHeight,normalizeWidth,CV_8UC3);
    for(int y_offset = 0; y_offset < normalizeHeight; y_offset++){
        for(int x_offset = 0; x_offset < normalizeWidth; x_offset++){
            normalizeImage.at<cv::Vec3b>(y_offset,x_offset) = rawImage.at<cv::Vec3b>(y_offset,x_offset);
        }
    }
    std::cout << "normalizeImage width: " << normalizeWidth << ", height: " << normalizeHeight << std::endl;

    cv::Mat features;
    //this->generateRGBFeatures(normalizeImage,features,patchSize);
    this->generateHistogramFeatures(normalizeImage,features,patchSize);

    cv::flann::Index myKdTree;
    myKdTree.build(features, cv::flann::KDTreeIndexParams(512));

    QDomDocument doc;
    QDomProcessingInstruction instruction = doc.createProcessingInstruction("xml","version=\"1.0\" encoding=\"UTF-8\"");
    doc.appendChild(instruction);
    QDomElement gexf = doc.createElement("gexf");{
        gexf.setAttribute("xmlns:viz","http:///www.gexf.net/1.1draft/viz");
        gexf.setAttribute("version","1.1");
        gexf.setAttribute("xmlns","http://www.gexf.net/1.1draft");
    }
    doc.appendChild(gexf);
    QDomElement meta = doc.createElement("meta");{
        QDateTime dt;
        QTime time;
        QDate date;
        dt.setTime(time.currentTime());
        dt.setDate(date.currentDate());
        QString currentDate = dt.toString("yyyy-MM-dd+hh:mm:ss");
        meta.setAttribute("lastmodifieddate",currentDate);
    }
    gexf.appendChild(meta);
    QDomElement creator = doc.createElement("creator");{
    }
    gexf.appendChild(creator);
    QDomElement graph = doc.createElement("graph");{
        graph.setAttribute("defaultedgetype","undirected");
        graph.setAttribute("idtype","string");
        graph.setAttribute("type","static");
    }
    gexf.appendChild(graph);
    QDomElement nodes = doc.createElement("nodes");{
        nodes.setAttribute("count",QString::number(features.rows,10));
    }
    graph.appendChild(nodes);
    QDomElement edges = doc.createElement("edges");{

    }
    graph.appendChild(edges);

    for(int i = 0; i < features.rows; i++){
        QDomElement node = doc.createElement("node");{
            node.setAttribute("id",QString::number(i,10));
            node.setAttribute("label",QString::number(i,10));
        }
        nodes.appendChild(node);
    }



    std::vector<bool> searched(features.rows,false);
    std::vector<std::vector<int> > indexCache;
    std::vector<std::vector<float> > distanceCache;
    int currentEdgeCount = 0;
    for(int i = 0; i < features.rows/*features.rows*/; i++){
        std::cout << i << "/" << features.rows << std::endl;
        cv::Mat index;
        cv::Mat distance;
        myKdTree.knnSearch(features.row(i),index,distance,10,cv::flann::SearchParams(512));
        std::vector<int> elemIndexCache;
        std::vector<float> elemDistanceCache;
        for(int j = 1; j < index.cols; j++){
            int targetIndex = index.at<int>(0,j);
            if(searched.at(targetIndex) == true){
                bool breakBit = false;
                std::vector<int> searchVector = indexCache.at(targetIndex);
                for(int searchIndex = 0; searchIndex<(int)searchVector.size(); searchIndex++){
                    if(searchVector.at(searchIndex) == i){
                        breakBit = true;
                        break;
                    }
                }
                if(breakBit == true){
                    break;
                }
            }
            QDomElement edge = doc.createElement("edge");{
                edge.setAttribute("id",QString::number(currentEdgeCount,10));
                currentEdgeCount++;
                edge.setAttribute("source",QString::number(i,10));
                edge.setAttribute("target",QString::number(targetIndex,10));
                elemIndexCache.push_back(targetIndex);
                edge.setAttribute("weight",QString("%1").arg(distance.at<float>(0,j)/patchSize));
                elemDistanceCache.push_back(distance.at<float>(0,j)/patchSize);
            }
            edges.appendChild(edge);
        }
        searched.at(i) = true;
        indexCache.push_back(elemIndexCache);
        distanceCache.push_back(elemDistanceCache);
    }
    edges.setAttribute("count",QString::number(currentEdgeCount,10));

    QFile file("output.xml");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate |QIODevice::Text))
        return ;
    QTextStream out(&file);
    out.setCodec("UTF-8");
    doc.save(out,4,QDomNode::EncodingFromTextStream);
    file.close();

    cv::Mat distanceMat = cv::Mat(features.rows,features.rows,CV_32F,INT32_MAX);
    for(int sourceIndex = 0; sourceIndex<features.rows; sourceIndex++){
        for(int i = 0; i < indexCache.at(sourceIndex).size(); i++){
            int targetIndex = indexCache.at(sourceIndex).at(i);
            float targetDistance = distanceCache.at(sourceIndex).at(i);

            distanceMat.at<float>(sourceIndex,targetIndex) = distanceMat.at<float>(targetIndex,sourceIndex) = targetDistance;
        }
    }
    for(int sourceIndex = 0; sourceIndex<features.rows; sourceIndex++){
        distanceMat.at<float>(sourceIndex,sourceIndex) = 0;
    }

    //std::cout  << distanceMat << std::endl;
    std::ofstream outputMat("outputmat.txt");
    outputMat << distanceMat;
    outputMat.close();
    std::ofstream outputFeatureMat("outputFeaturemat.txt");
    outputFeatureMat << features;
    outputFeatureMat.close();


    cv::imshow("rawImage",rawImage);
    cv::imshow("normalizeImage",normalizeImage);

}

//这里传入的图片已经被规则化，能被patchSize整除
void GenerateNeighborXML::generateRGBFeatures(const cv::Mat& sourceImage, cv::Mat& features, int patchSize){
    features = cv::Mat(sourceImage.rows/patchSize*sourceImage.cols/patchSize, patchSize*patchSize*3, CV_32F);
    std::cout << "Generate feature width: " << features.cols << ", height: " << features.rows << std::endl;

    for(int y_offset = 0; y_offset < sourceImage.rows; y_offset+=patchSize){
        for(int x_offset = 0; x_offset < sourceImage.cols; x_offset+=patchSize){
            for(int y_patch_offset = 0; y_patch_offset < patchSize; y_patch_offset++){
                for(int x_patch_offset = 0; x_patch_offset < patchSize; x_patch_offset++){
                    features.at<float>(y_offset/patchSize*sourceImage.cols/patchSize+x_offset/patchSize,(y_patch_offset*patchSize+x_patch_offset)*3+0) = (float)sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[0];
                    features.at<float>(y_offset/patchSize*sourceImage.cols/patchSize+x_offset/patchSize,(y_patch_offset*patchSize+x_patch_offset)*3+1) = (float)sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[1];
                    features.at<float>(y_offset/patchSize*sourceImage.cols/patchSize+x_offset/patchSize,(y_patch_offset*patchSize+x_patch_offset)*3+2) = (float)sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[2];
                }
            }
        }
    }
}

void GenerateNeighborXML::generateHistogramFeatures(const cv::Mat& sourceImage, cv::Mat& features, int patchSize){
    const int binCount = 8;
    const int colorStep = 256/binCount;

    features = cv::Mat(sourceImage.rows/patchSize*sourceImage.cols/patchSize,binCount*3,CV_32F);
    for(int y_offset = 0; y_offset < sourceImage.rows; y_offset+=patchSize){
        for(int x_offset = 0; x_offset < sourceImage.cols; x_offset+=patchSize){
            std::cout <<y_offset << " " << x_offset << "start"<<std::endl;
            std::vector<int> blueBins(binCount,0);
            std::vector<int> greenBins(binCount,0);
            std::vector<int> redBins(binCount,0);
            for(int y_patch_offset = 0; y_patch_offset < patchSize; y_patch_offset++){
                for(int x_patch_offset = 0; x_patch_offset < patchSize; x_patch_offset++){
                    std::cout << "blue:" << (int)sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[0] << std::endl;
                    std::cout << "para: " << sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[0]/colorStep;
                    std::cout << " " << sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[1]/colorStep;
                    std::cout << " " <<     sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[2]/colorStep << std::endl;
                    blueBins.at(sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[0]/colorStep)++;
                    greenBins.at(sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[1]/colorStep)++;
                    redBins.at(sourceImage.at<cv::Vec3b>(y_offset+y_patch_offset,x_offset+x_patch_offset)[2]/colorStep)++;
                }
            }
            for(int elem : blueBins){
                std::cout << elem << " ";
            }
            std::cout << std::endl;
            std::cout <<y_offset << " " << x_offset << "mid"<<std::endl;
            for(int i = 0; i < binCount; i++){
                //std::cout << "para: " << y_offset/patchSize*sourceImage.cols<< " " << binCount*0+i << std::endl;
                features.at<float>(y_offset/patchSize*sourceImage.cols/patchSize+x_offset/patchSize, binCount*0+i) = (float)blueBins.at(i);
                features.at<float>(y_offset/patchSize*sourceImage.cols/patchSize+x_offset/patchSize, binCount*1+i) = (float)greenBins.at(i);
                features.at<float>(y_offset/patchSize*sourceImage.cols/patchSize+x_offset/patchSize, binCount*2+i) = (float)redBins.at(i);
            }
            std::cout <<y_offset << " " << x_offset << "end"<<std::endl;
        }
    }
    std::cout <<"generateHistogramFeatures"<<std::endl;
}
