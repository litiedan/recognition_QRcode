#include "zbar.h"      
#include "cv.h"      
#include "highgui.h"      
#include <iostream>      
 
using namespace std;      
using namespace zbar;  //添加zbar名称空间    
using namespace cv;      
 
int main(int argc,char*argv[])    
{  
	Mat imageSource=imread(argv[1],0);	
	Mat image;
	imageSource.copyTo(image);
	GaussianBlur(image,image,Size(3,3),0);  //滤波
	threshold(image,image,100,255,CV_THRESH_BINARY);  //二值化
	imshow("二值化",image);	
	Mat element=getStructuringElement(2,Size(7,7));	 //膨胀腐蚀核
	//morphologyEx(image,image,MORPH_OPEN,element);	
	for(int i=0;i<10;i++)
	{
		erode(image,image,element);
		i++;
	}	
	imshow("腐蚀s",image);
	Mat image1;
	erode(image,image1,element);
	image1=image-image1;
	imshow("边界",image1);
	//寻找直线 边界定位也可以用findContours实现
	vector<Vec2f>lines;
	HoughLines(image1,lines,1,CV_PI/150,250,0,0);
	Mat DrawLine=Mat::zeros(image1.size(),CV_8UC1);
	for(int i=0;i<lines.size();i++)
	{
		float rho=lines[i][0];
		float theta=lines[i][1];
		Point pt1,pt2;
		double a=cos(theta),b=sin(theta);
		double x0=a*rho,y0=b*rho;
		pt1.x=cvRound(x0+1000*(-b));
		pt1.y=cvRound(y0+1000*a);
		pt2.x=cvRound(x0-1000*(-b));
		pt2.y=cvRound(y0-1000*a);
		line(DrawLine,pt1,pt2,Scalar(255),1,CV_AA);
	}
	imshow("直线",DrawLine);
	Point2f P1[4];
	Point2f P2[4];
	vector<Point2f>corners;
	goodFeaturesToTrack(DrawLine,corners,4,0.1,10,Mat()); //角点检测
	for(int i=0;i<corners.size();i++)
	{
		circle(DrawLine,corners[i],3,Scalar(255),3);
		P1[i]=corners[i];		
	}
	imshow("交点",DrawLine);
	int width=P1[1].x-P1[0].x;
	int hight=P1[2].y-P1[0].y;
	P2[0]=P1[0];
	P2[1]=Point2f(P2[0].x+width,P2[0].y);
	P2[2]=Point2f(P2[0].x,P2[1].y+hight);
	P2[3]=Point2f(P2[1].x,P2[2].y);
	Mat elementTransf;
	elementTransf=	getAffineTransform(P1,P2);
	warpAffine(imageSource,imageSource,elementTransf,imageSource.size(),1,0,Scalar(255));
	imshow("校正",imageSource);	
	//Zbar二维码识别
	ImageScanner scanner;      
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1); 
	int width1 = imageSource.cols;      
	int height1 = imageSource.rows;      
	uchar *raw = (uchar *)imageSource.data;         
	Image imageZbar(width1, height1, "Y800", raw, width * height1);        
	scanner.scan(imageZbar); //扫描条码    
	Image::SymbolIterator symbol = imageZbar.symbol_begin();  
	if(imageZbar.symbol_begin()==imageZbar.symbol_end())  
	{  
		cout<<"查询条码失败，请检查图片！"<<endl;  
	}  
	for(;symbol != imageZbar.symbol_end();++symbol)    
	{      
		cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;    
		cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;       
	}      
	namedWindow("Source Window",0);
	imshow("Source Window",imageSource);        
	waitKey();    
	imageZbar.set_data(NULL,0);  
	return 0;  
}

