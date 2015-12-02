#include "readsuperpixeldat.h"
#include <cmath>
#include <fstream>

ReadSuperPixelDat::ReadSuperPixelDat()
{

}

#include <string>
#include "/home/netbeen/桌面/SLICSuperpixels_VC2008_SLICO_15Jun2013/SLICSuperpixels/SLIC.h"

void ReadSuperPixelDat::authorSegment()
{
    const int width = this->image.cols;
    const int height = this->image.rows;
    // unsigned int (32 bits) to hold a pixel in ARGB format as follows:
    // from left to right,
    // the first 8 bits are for the alpha channel (and are ignored)
    // the next 8 bits are for the red channel
    // the next 8 bits are for the green channel
    // the last 8 bits are for the blue channel
    //unsigned int* pbuff = new UINT[sz];
    ReadImage(pbuff, width, height);//YOUR own function to read an image into the ARGB format

    //----------------------------------
    // Initialize parameters
    //----------------------------------
    int k = 2000;//Desired number of superpixels.
    double m = 40;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
    int* klabels = new int[width*height];
    int numlabels(0);
    string filename = "yourfilename.jpg";
    string savepath = "yourpathname";
    //----------------------------------
    // Perform SLIC on the image buffer
    //----------------------------------
    SLIC segment;
    segment.PerformSLICO_ForGivenK(pbuff, width, height, klabels, numlabels, k, m);
    // Alternately one can also use the function PerformSLICO_ForGivenStepSize() for a desired superpixel size
    //----------------------------------
    // Save the labels to a text file
    //----------------------------------
    segment.SaveSuperpixelLabels(klabels, width, height, filename, savepath);
    //----------------------------------
    // Draw boundaries around segments
    //----------------------------------
    segment.DrawContoursAroundSegments(pbuff, klabels, width, height, 0xff0000);
    //----------------------------------
    // Save the image with segment boundaries.
    //----------------------------------
    SaveSegmentedImageFile(pbuff, width, height);//YOUR own function to save an ARGB buffer as an image
    //----------------------------------
    // Clean up
    //----------------------------------
    if(pbuff) delete [] pbuff;
    if(klabels) delete [] klabels;
}


void ReadSuperPixelDat::main(){
    this->image = cv::imread("/home/netbeen/桌面/周叔项目/stone.png");
    this->size = this->image.cols * this->image.rows;
    std::cout << this->size << std::endl;


    this->authorSegment();
    this->authorRead();
    std::ofstream labelsFile("labels.txt");
    for(int i = 0; i < this->size; i++){
        std::cout << i << std::endl;
        labelsFile << labels[i] << std::endl;
    }
    labelsFile.close();


    this->analyseLabelFile();

    std::cout << "Done!" << std::endl;
}

void ReadSuperPixelDat::ReadImage(unsigned int* pbuff, int width,int height){//YOUR own function to read an image into the ARGB format
    std::cout << "ReadImage start" << std::endl;
    const int sz = this->size;
    this->pbuff = new unsigned int[sz];
    std::cout << "pbuff's address = " <<(pbuff) << std::endl;

    unsigned int* ptr = this->pbuff;
    for(int y_offset = 0; y_offset < this->image.rows; y_offset++){
        for(int x_offset = 0; x_offset < this->image.cols; x_offset++){
            cv::Vec3b color = this->image.at<cv::Vec3b>(y_offset,x_offset);
            unsigned int tempResult = 1;
            tempResult += color[0];
            tempResult += color[1]*std::pow(2,8);
            tempResult += color[2]*std::pow(2,16);
            (*ptr) = (unsigned int)tempResult;
            ptr++;
        }
    }

    std::cout << "ReadImage end" << std::endl;
}


void ReadSuperPixelDat::SaveSegmentedImageFile(unsigned int* pbuff, int width,int height){//YOUR own function to save an ARGB buffer as an image
    std::cout << "SaveSegmentedImageFile start" << std::endl;
    this->result = cv::Mat(this->image.size(), CV_8UC3);

    unsigned int* ptr = this->pbuff;
    for(int y_offset = 0; y_offset < this->image.rows; y_offset++){
        for(int x_offset = 0; x_offset < this->image.cols; x_offset++){
            std::cout << y_offset << " " << x_offset << std::endl;
            unsigned int tempResult1 = *ptr;
            int blue = tempResult1%(int)std::pow(2,8);
            int green = (tempResult1/(int)std::pow(2,8))%(int)std::pow(2,8);
            int red = (tempResult1/(int)std::pow(2,16))%(int)std::pow(2,8);
            this->result.at<cv::Vec3b>(y_offset,x_offset) = cv::Vec3b(blue,green,red);
            ptr++;
        }
    }
    cv::imwrite("result.png",this->result);

    std::cout << "SaveSegmentedImageFile end" << std::endl;
}

void ReadSuperPixelDat::authorRead(){
    FILE* pf = fopen("output.dat", "r");
    int sz = this->size;
    int* vals = new int[sz];
    int elread = fread((char*)vals, sizeof(int), sz, pf);
    for( int j = 0; j < this->image.rows; j++ )
    {
        for( int k = 0; k < this->image.cols; k++ )
        {
            int i = j*this->image.cols+k;
            labels[i] = vals[i];
        }
    }
    delete [] vals;
    fclose(pf);
}


void ReadSuperPixelDat::analyseLabelFile(){
    std::ifstream labelFile("labels.txt");
    std::vector<int> labels;
    int minLabel = INT32_MAX;
    int maxLabel= INT32_MIN;
    for(int i = 0; i < this->size; i++){
        int tempLabel;
        labelFile >> tempLabel;
        labels.push_back(tempLabel);
        minLabel = std::min(minLabel, tempLabel);
        maxLabel = std::max(maxLabel, tempLabel);
    }
    labelFile.close();
    std::cout << minLabel << " " << maxLabel << std::endl;
//    for(int i = 0; i < this->size; i++){
//        std::cout << i <<" " << labels.at(i) << std::endl;
//    }

    std::vector<std::vector< std::pair<int,int> > > label2Coodinates(maxLabel+1);
    std::vector<int> label2Count(maxLabel+1,0);
    cv::Mat coodinate2Label = cv::Mat(this->image.rows, this->image.cols, CV_32S);

    for(int i = 0; i < this->size; i++){
        int y_offset = i/this->image.cols;
        int x_offset = i%this->image.cols;

        coodinate2Label.at<int>(y_offset,x_offset) = labels.at(i);      //CHECKED
        label2Count.at(labels.at(i))++;     //CHECKED
        label2Coodinates.at(labels.at(i)).push_back(std::pair<int,int>(y_offset,x_offset));     //CHECKED
    }
    //std::cout << coodinate2Label << std::endl;
//    for(int elem : label2Count){
//        std::cout << elem << std::endl;
//    }
//    for(std::vector< std::pair<int,int> > elem : label2Coodinates){
//        std::cout << elem.size() << std::endl;
//    }

    this->mask = cv::imread("/home/netbeen/桌面/周叔项目/mask.png",cv::IMREAD_GRAYSCALE);

    this->maskID2Coodinates = std::vector<std::vector< std::pair<int,int> > >();
    this->maskDFS = mask.clone();

    this->searchWriteArea();

    //std::cout << this->maskID2Coodinates.size() << std::endl;

    std::ofstream analyseResult("analyseResult.txt");
    analyseResult << "#labels" << std::endl;
    analyseResult << label2Coodinates.size() << std::endl;
    for(int i = 0 ; i < label2Coodinates.size(); i++){
        analyseResult << i << "\t" << label2Coodinates.at(i).size() <<"\t";
        for(std::pair<int,int> elem : label2Coodinates.at(i)){
            analyseResult << elem.first << "," << elem.second << "\t";
        }
        analyseResult << std::endl;
    }
    analyseResult << "#masks" << std::endl;
    analyseResult << maskID2Coodinates.size() << std::endl;

    const float THRESHOLD = 0.75;

    for(int i = 0 ; i < maskID2Coodinates.size(); i++){
        analyseResult << i << "\t" ;
        std::vector<int> coverLabelCount(maxLabel+1,0);
        for(std::pair<int,int> elem : maskID2Coodinates.at(i)){
            int currentLabel = coodinate2Label.at<int>(elem.second, elem.first);
            coverLabelCount.at(currentLabel)++;
        }
        assert(coverLabelCount.size() == label2Count.size());
        std::vector<int> result;
        for(int j = 0; j < coverLabelCount.size(); j++){
            if((float)coverLabelCount.at(j) / (float)label2Count.at(j) > THRESHOLD){
                result.push_back(j);
            }
        }
        analyseResult << result.size() << "\t" ;
        for(int elem : result){
            analyseResult << elem << "\t" ;
        }
        analyseResult << std::endl;
    }




    analyseResult.close();
}

void ReadSuperPixelDat::searchWriteArea(){
    for(int y_offset = 0; y_offset < this->maskDFS.rows; y_offset++){
        for(int x_offset = 0; x_offset < this->maskDFS.cols; x_offset++){
            if(this->maskDFS.at<uchar>(y_offset,x_offset) == 255){
                std::cout << "Find write area: " << y_offset << " " << x_offset << std::endl;
                this->coordinates = std::vector< std::pair<int,int> >();
                this->dfs(std::pair<int,int>(y_offset,x_offset));
                this->maskID2Coodinates.push_back(this->coordinates);
                std::cout << "This area of mask has point count: " << this->coordinates.size() << std::endl;
                //return;
            }
        }
    }
}

void ReadSuperPixelDat::dfs(std::pair<int,int> coordinate){
    this->coordinates.push_back(coordinate);
    this->maskDFS.at<uchar>(coordinate.first,coordinate.second) = 127;
    if(this->maskDFS.at<uchar>(coordinate.first,coordinate.second+1) == 255){
        this->dfs(std::pair<int,int>(coordinate.first,coordinate.second+1));
    }
    if(this->maskDFS.at<uchar>(coordinate.first+1,coordinate.second) == 255){
        this->dfs(std::pair<int,int>(coordinate.first+1,coordinate.second));
    }
    if(this->maskDFS.at<uchar>(coordinate.first,coordinate.second-1) == 255){
        this->dfs(std::pair<int,int>(coordinate.first,coordinate.second-1));
    }
    if(this->maskDFS.at<uchar>(coordinate.first-1,coordinate.second) == 255){
        this->dfs(std::pair<int,int>(coordinate.first-1,coordinate.second));
    }
}
