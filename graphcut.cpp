#include "graphcut.h"

GraphCut::GraphCut()
{

}

void GraphCut::main(){
    cv::Mat rawImage = cv::imread("/home/netbeen/桌面/周叔项目/trg (2).png");
    cv::Mat userMark = cv::imread("/home/netbeen/桌面/周叔项目/trg_label-mark.png", cv::IMREAD_GRAYSCALE);

    if(this->checkUserMarkValid(userMark) == false){
        std::cout << "checkUserMarkValid false!" <<std::endl;
        exit(1);
    }else{
        std::cout << "checkUserMarkValid true!" <<std::endl;
    }

    cv::imshow("rawImage",rawImage);
    cv::waitKey();
}

void GraphCut::GridGraph_Individually(int width,int height,int num_pixels,int num_labels)
{

    int *result = new int[num_pixels];   // stores result of optimization

    try{
        GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

        std::vector<uchar>::iterator it;
        // first set up data costs individually
        for ( int i = 0; i < num_pixels; i++ ){
            int currentRow = i/this->rawImage.cols;
            int currentCol = i%this->rawImage.cols;
            uchar currentGrayValue = this->initGuessGray.at<uchar>(currentRow,currentCol);
            it = std::find(this->label2GrayValue.begin(), this->label2GrayValue.end(), currentGrayValue);
            int label = it - this->label2GrayValue.begin();
            for(int labelIndex = 0; labelIndex < this->CLASS_NUMBER; labelIndex++){
                if(label == labelIndex){
                    gc->setDataCost(i,labelIndex,1);
                }else{
                    gc->setDataCost(i,labelIndex,2);
                }
                //gc->setDataCost(i,labelIndex,1);
            }
        }

        // next set up smoothness costs individually
        for ( int l1 = 0; l1 < num_labels; l1++ ){
            for (int l2 = 0; l2 < num_labels; l2++ ){
                int cost = (l1-l2)*(l1-l2) <= 64  ? 3*(l1-l2)*(l1-l2):64;
                gc->setSmoothCost(l1,l2,cost);
                //gc->setSmoothCost(l1,l2,1);
            }
        }

        printf("\nBefore optimization energy is %d",gc->compute_energy());
        gc->expansion(5);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
        printf("\nAfter optimization energy is %d\n",gc->compute_energy());



        for ( int  i = 0; i < num_pixels; i++ ){
            result[i] = gc->whatLabel(i);
            this->resultLabelGray.at<uchar>(i/this->resultLabelGray.cols, i%this->resultLabelGray.cols) = this->label2GrayValue.at(result[i]);
        }


        delete gc;
    }
    catch (GCException e){
        e.Report();
    }

    delete [] result;
}



/**
 * @brief GraphCut::main2
 * @brief 使用预先进行分类的分类图作为最初猜想
 */
void GraphCut::main2(){
    this->rawImage = cv::imread("/home/netbeen/桌面/周叔项目/trg (2).png");
    //cv::resize(this->rawImage,this->rawImage,cv::Size(this->rawImage.rows/this->scaleFactor,this->rawImage.cols/this->scaleFactor));
    this->initGuessGray = cv::imread("/home/netbeen/桌面/周叔项目/trg_label.png", cv::IMREAD_GRAYSCALE);
    this->resultLabelGray = cv::Mat(this->rawImage.rows,this->rawImage.cols,CV_8U);


    if(this->checkUserMarkValid(this->initGuessGray) == false){
        std::cout << "checkUserMarkValid false!" <<std::endl;
        exit(1);
    }else{
        std::cout << "checkUserMarkValid true!" <<std::endl;
    }

    int num_pixels = this->rawImage.cols*this->rawImage.rows;

    // smoothness and data costs are set up one by one, individually
    this->GridGraph_Individually(this->rawImage.cols,this->rawImage.rows,num_pixels,this->CLASS_NUMBER);

    /*this->initGuessGray = this->resultLabelGray;
    this->GridGraph_Individually(this->rawImage.cols,this->rawImage.rows,num_pixels,this->CLASS_NUMBER);*/

    cv::imshow("resultLabelGray",this->resultLabelGray);
    cv::imshow("initGuessGray",this->initGuessGray);
    cv::imshow("rawImage",this->rawImage);
    cv::waitKey();
}




bool GraphCut::checkUserMarkValid(const cv::Mat& userMark){
    for(int offset_y = 0; offset_y < userMark.rows; offset_y++){
        for(int offset_x = 0; offset_x < userMark.cols; offset_x++){
            uchar currentMark = userMark.at<uchar>(offset_y,offset_x);
            if(currentMark == 0){
                continue;
            }
            std::vector<uchar>::iterator it = std::find(this->label2GrayValue.begin(),this->label2GrayValue.end(),currentMark);
            if(it == this->label2GrayValue.end()){
                this->label2GrayValue.push_back(currentMark);
                std::cout << "Find new mark: " << static_cast<int>(currentMark) << std::endl;
                if(this->label2GrayValue.size() > this->CLASS_NUMBER){
                    return false;
                }
            }
        }
    }
    if(this->label2GrayValue.size() == this->CLASS_NUMBER){
        return true;
    }else{
        return false;
    }
}
