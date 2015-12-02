#include "kmeans.h"
#include <fstream>

KMeans::KMeans()
{

}

void KMeans::main(){
    cv::Mat rawImage = cv::imread("/home/netbeen/桌面/周叔项目/stone.png");


    cv::Mat mat_sample(rawImage.rows*rawImage.cols, 3, CV_32F);
    for(int y_offset = 0; y_offset < rawImage.rows; y_offset++){
        for(int x_offset = 0; x_offset < rawImage.cols; x_offset++){
            mat_sample.at<float>(y_offset*rawImage.cols+x_offset,0) = (float)rawImage.at<cv::Vec3b>(y_offset,x_offset)[0];
            mat_sample.at<float>(y_offset*rawImage.cols+x_offset,1) = (float)rawImage.at<cv::Vec3b>(y_offset,x_offset)[1];
            mat_sample.at<float>(y_offset*rawImage.cols+x_offset,2) = (float)rawImage.at<cv::Vec3b>(y_offset,x_offset)[2];
        }
    }
    std::cout << "mat_sample done" << std::endl;
    const int kMeansItCount = 10;
    const int kMeansType = cv::KMEANS_PP_CENTERS;
    cv::Mat label;

    cv::kmeans( mat_sample, 32, label,cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    std::ofstream output1("32output.label");
    output1 << label;
    output1.close();

    cv::kmeans( mat_sample, 64, label,cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    std::ofstream output2("64output.label");
    output2 << label;
    output2.close();

    cv::kmeans( mat_sample, 128, label,cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    std::ofstream output3("128output.label");
    output3 << label;
    output3.close();

    cv::kmeans( mat_sample, 256, label,cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    std::ofstream output4("256output.label");
    output4 << label;
    output4.close();
}
