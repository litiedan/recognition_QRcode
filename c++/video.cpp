//运行以下指令进行编译
//g++ video.cpp `pkg-config --cflags --libs opencv` -o video
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std; 
int main()
{
	//读取视频或摄像头
	//VideoCapture capture(0);//USB摄像头
 	VideoCapture capture("rtsp://admin:admin@192.168.1.111//stream1");//ip摄像头
	while (true)
	{
		Mat frame;
		capture >> frame;
		cout<<frame;
		imshow("读取视频", frame);
		waitKey(30);	//延时30
	}
	return 0;

}
