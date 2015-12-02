#include "mygabor.h"
#include "utils.h"



using namespace cv;
using namespace std;

MyGabor::MyGabor()
{

}





void MyGabor::main(){

    int type = 0;

    Mat image = imread("/home/netbeen/桌面/img5.jpg"); // Read the file
    if(! image.data ){
        cout << "Could not open or find the image" << std::endl ;
        exit(1);
    }
    //cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
    resize(image,image,cv::Size(image.cols/2.5,image.rows/2.5));

    if(image.channels() == 3){
        cv::Mat blueValue = cv::Mat(image.rows, image.cols, CV_8U);
        cv::Mat greenValue = cv::Mat(image.rows, image.cols, CV_8U);
        cv::Mat redValue = cv::Mat(image.rows, image.cols, CV_8U);

        for(int offset_y = 0; offset_y < image.rows; offset_y++){
            for(int offset_x = 0; offset_x < image.cols; offset_x++){
                blueValue.at<uchar>(offset_y,offset_x) = image.at<cv::Vec3b>(offset_y,offset_x)[0];
                greenValue.at<uchar>(offset_y,offset_x) = image.at<cv::Vec3b>(offset_y,offset_x)[1];
                redValue.at<uchar>(offset_y,offset_x) = image.at<cv::Vec3b>(offset_y,offset_x)[2];
            }
        }
        cv::imshow("blueValue",blueValue);
        cv::imshow("greenValue",greenValue);
        cv::imshow("redValue",redValue);

        Mat blue_filterd_image = this->gabor_filter(blueValue, type);
        Mat green_filterd_image = this->gabor_filter(greenValue, type);
        Mat red_filterd_image = this->gabor_filter(redValue, type);

        //std::cout << blue_filterd_image.row(0) << std::endl;

        cv::Mat blue_features = this->generateFeature(blue_filterd_image);
        cv::Mat green_features = this->generateFeature(green_filterd_image);
        cv::Mat red_features = this->generateFeature(red_filterd_image);

        //std::cout << blue_features.row(0) << std::endl;

        cv::Mat features = cv::Mat(blue_features.rows, blue_features.cols*3, blue_features.type());
        for(int offset_y = 0; offset_y < blue_features.rows; offset_y++){
            for(int offset_x = 0; offset_x < blue_features.cols; offset_x++){
                int step = blue_features.cols;
                features.at<float>(offset_y,offset_x+step*0) = blue_features.at<float>(offset_y,offset_x);
                features.at<float>(offset_y,offset_x+step*1) = green_features.at<float>(offset_y,offset_x);
                features.at<float>(offset_y,offset_x+step*2) = red_features.at<float>(offset_y,offset_x);
            }
        }



        Utils::doDensityPeakAndShow(features, image,6);

    }else{
        Mat filterd_image = this->gabor_filter(image, type);
        imshow("filtered image", filterd_image);


        cv::Mat features = this->generateFeature(filterd_image);

        Utils::doDensityPeakAndShow(features, image,2);
    }






    waitKey(0);

    return;
}


Mat MyGabor::getMyGabor(int width, int height, int U, int V, double Kmax, double f,double sigma, int ktype, const string kernel_name)
{
    int half_width = width / 2;
    int half_height = height / 2;
    double Qu = PI*U/8;
    double sqsigma = sigma*sigma;
    double Kv = Kmax/pow(f,V);
    double postmean = exp(-sqsigma/2);

    Mat kernel_re(width, height, ktype);
    Mat kernel_im(width, height, ktype);
    Mat kernel_mag(width, height, ktype);

    double tmp1, tmp2, tmp3;
    for(int j = -half_height; j <= half_height; j++){
        for(int i = -half_width; i <= half_width; i++){
            tmp1 = exp(-(Kv*Kv*(j*j+i*i))/(2*sqsigma));
            tmp2 = cos(Kv*cos(Qu)*i + Kv*sin(Qu)*j) - postmean;
            tmp3 = sin(Kv*cos(Qu)*i + Kv*sin(Qu)*j);

            if(ktype == CV_32F)
                kernel_re.at<float>(j+half_height, i+half_width) =
                        (float)(Kv*Kv*tmp1*tmp2/sqsigma);
            else
                kernel_re.at<double>(j+half_height, i+half_width) =
                        (double)(Kv*Kv*tmp1*tmp2/sqsigma);

            if(ktype == CV_32F)
                kernel_im.at<float>(j+half_height, i+half_width) =
                        (float)(Kv*Kv*tmp1*tmp3/sqsigma);
            else
                kernel_im.at<double>(j+half_height, i+half_width) =
                        (double)(Kv*Kv*tmp1*tmp3/sqsigma);
        }
    }

    magnitude(kernel_re, kernel_im, kernel_mag);

    if(kernel_name.compare("real") == 0)
        return kernel_re;
    else if(kernel_name.compare("imag") == 0)
        return kernel_im;
    else{
        printf("Invalid kernel name!\n");
        return kernel_mag;
    }
}

void MyGabor::construct_gabor_bank()
{
    const int kernel_size = 69;
    double Kmax = PI/2;
    double f = sqrt(2.0);
    double sigma = 2*PI;
    int U = 0;
    int V = 0;
    int GaborH = kernel_size;
    int GaborW = kernel_size;
    int UStart = 0, UEnd = 8;
    int VStart = -1, VEnd = 4;

    Mat kernel;
    Mat totalMat;
    for(U = UStart; U < UEnd; U++){
        Mat colMat;
        for(V = VStart; V < VEnd; V++){
            kernel = getMyGabor(GaborW, GaborH, U, V,
                                Kmax, f, sigma, CV_64F, "real");

            //show gabor kernel
            normalize(kernel, kernel, 0, 1, CV_MINMAX);
            printf("U%dV%d\n", U, V);

            if(V == VStart)
                colMat = kernel;
            else
                vconcat(colMat, kernel, colMat);
        }
        if(U == UStart)
            totalMat = colMat;
        else
            hconcat(totalMat, colMat, totalMat);
    }

    imshow("gabor bank", totalMat);
    waitKey(0);
}

Mat MyGabor::gabor_filter(Mat& img, int type)
{
    const int kernel_size = 7; // should be odd
    // variables for gabor filter
    double Kmax = PI/2;
    double f = sqrt(2.0);
    double sigma = 2*PI;
    int U = 7;
    int V = 4;
    int GaborH = kernel_size;
    int GaborW = kernel_size;
    int UStart = 0, UEnd = 8;
    int VStart = -1, VEnd = 4;

    //
    Mat kernel_re, kernel_im;
    Mat dst_re, dst_im, dst_mag;

    // variables for filter2D
    Point archor(-1,-1);
    int ddepth = CV_64F;//CV_64F
    double delta = 0;

    // filter image with gabor bank
    Mat totalMat, totalMat_re, totalMat_im;
    for(U = UStart; U < UEnd; U++){
        Mat colMat, colMat_re, colMat_im;
        for(V = VStart; V < VEnd; V++){
            kernel_re = getMyGabor(GaborW, GaborH, U, V,
                                   Kmax, f, sigma, CV_64F, "real");
            kernel_im = getMyGabor(GaborW, GaborH, U, V,
                                   Kmax, f, sigma, CV_64F, "imag");

            filter2D(img, dst_re, ddepth, kernel_re);
            filter2D(img, dst_im, ddepth, kernel_im);

            dst_mag.create(img.rows, img.cols, CV_64FC1);
            magnitude(Mat_<float>(dst_re),Mat_<float>(dst_im),
                      dst_mag);

            //show gabor kernel
            normalize(dst_mag, dst_mag, 0, 1, CV_MINMAX);
            normalize(dst_re, dst_re, 0, 1, CV_MINMAX);
            normalize(dst_im, dst_im, 0, 1, CV_MINMAX);


            if(V == VStart){
                colMat = dst_mag;
                colMat_re = dst_re;
                colMat_im = dst_im;
            }
            else{
                vconcat(colMat, dst_mag, colMat);
                vconcat(colMat_re, dst_re, colMat_re);
                vconcat(colMat_im, dst_im, colMat_im);
            }
        }
        if(U == UStart){
            totalMat = colMat;
            totalMat_re = colMat_re;
            totalMat_im = colMat_im;
        }
        else{
            hconcat(totalMat, colMat, totalMat);
            hconcat(totalMat_re, colMat_re, totalMat_re);
            hconcat(totalMat_im, colMat_im, totalMat_im);
        }
    }

    // return
    switch(type){
        case 0:
            return totalMat;
        case 1:
            return totalMat_re;
        case 2:
            return totalMat_im;
        default:
            return totalMat;
    }
}




cv::Mat MyGabor::generateFeature(const cv::Mat& gaborImg){

    const int imgWidth = gaborImg.cols/8;
    const int imgHeight = gaborImg.rows/5;
    const int featureDimension = 8*5;

    cv::Mat features = cv::Mat(imgWidth*imgHeight, featureDimension, CV_32F);

    for(int offset_y = 0; offset_y < imgHeight; offset_y++){
        for(int offset_x = 0; offset_x < imgWidth; offset_x++){
            for(int dimension_index = 0; dimension_index < featureDimension; dimension_index++){
                const int gaborFeatureX = dimension_index%8;
                const int gaborFeatureY = dimension_index/8;

                float targetGrayValue = gaborImg.at<float>(gaborFeatureY*imgHeight + offset_y,  gaborFeatureX*imgWidth+ offset_x);
                features.at<float>(offset_y * imgWidth + offset_x,  gaborFeatureY*8+gaborFeatureX) = targetGrayValue;
            }
        }
    }

    return features;
}
