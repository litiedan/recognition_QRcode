//g++ show.cpp `pkg-config --cflags --libs opencv` -o lena
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;



int main( )
{
    Mat image;
    image = imread("/home/mr/cam_file/recognition_QRcode/rikirobot101.png", 1 );//图像与.cpp文件在同一目录下



    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }



    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);



    waitKey(0);



    return 0;
}
