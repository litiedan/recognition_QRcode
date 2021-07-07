import cv2
import numpy as np
import copy
import math


def detecte(image):
    '''提取所有轮廓'''
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#// Convert Image captured from Image Input to GrayScale	
    canny = cv2.Canny(gray, 100, 200,3)#Apply Canny edge detection on the gray image
    contours,hierachy=cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#Find contours with hierarchy
    return contours,hierachy
    #coutour是一个list，每个元素都是一个轮廓（彻底围起来算一个轮廓），用numpy中的ndarray表示。
    #hierarchy也是一个ndarry，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，
    #分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
def conter(image):
    contours,hierarchy=detecte(image)
    # print(len(contours))
    centers = {}
    M = {}
    # lista = [0,1,2,float("nan"),4]
    # print(lista[3])
    for i in range(len(contours)):
        M[i] = cv2.moments(contours[i])
        if(M[i]["m00"] == 0):
            centers[i] = (float("nan"), float("nan"))
        else:
            centers[i] = (float(M[i]["m10"] / M[i]["m00"]), float(M[i]["m01"] / M[i]["m00"]))
        # print(centers[i])

    mark = 0
    hierarchy = hierarchy[0]
    # print(hierarchy[0])

    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c+1
        if hierarchy[k][2] != -1:
            c = c+1
        if c >= 5:
            if mark == 0 :A = i
            elif mark == 1 :B = i
            elif mark == 2 :C = i
            mark = mark + 1
    # print(mark)
    realPosition = [-1,-1]
    RotationAngle = -1
    if mark >=3:
        AB = cv_distance(centers[A],centers[B])
        BC = cv_distance(centers[B],centers[C])
        CA = cv_distance(centers[C],centers[A])
        # print(AB)
        # print(BC)
        # print(CA)
        # print("three control points:",centers[A],centers[B],centers[C])

        if AB > BC and AB > CA:
            outlier = C
            median1 = A
            median2 = B
        elif CA > AB and CA > BC:
            outlier = B
            median1 = A
            median2 = C
        elif BC > AB and BC > CA:
            outlier = A
            median1 = B
            median2 = C
        top = outlier
        # print("top point:",centers[top])
        

        CentralPoint_x = (centers[median1][0]+centers[median2][0])/2
        CentralPoint_y = (centers[median1][1]+centers[median2][1])/2
        CentralPoint = [CentralPoint_x,CentralPoint_y]
        # print("Central point:",CentralPoint)

        realPosition_x =  CentralPoint_x / 1920 * 4.20
        realPosition_y =  CentralPoint_y / 1080 * 2.40
        realPosition = [realPosition_x,realPosition_y]
        # print("real point:",realPosition)

        # 	//定义一个位于二维码左上方各200个像素的的DefaultTopPoint，
        # //当二维码旋转0度时，top点一定在此DefaultTopPoint与二维码质心CentralPoint的连线上
        DefaultTopPoint_x = CentralPoint_x + 200
        DefaultTopPoint_y = CentralPoint_y - 200
        DefaultTopPoint = [DefaultTopPoint_x,DefaultTopPoint_y]

        Sdirection = (DefaultTopPoint_x - centers[top][0]) * (CentralPoint_y - centers[top][1]) -  (DefaultTopPoint_y - centers[top][1]) * (CentralPoint_x - centers[top][0])
        
        if Sdirection == 0:
            if centers[top][0]<CentralPoint_x: RotationAngle = 0#关键点的x坐标小于旋转中心的x坐标
            else:RotationAngle = 180
        else: 
            # //通过余弦定理，已知三边求角度
            aa = cv_distance(DefaultTopPoint,centers[top])
            bb = cv_distance(centers[top],CentralPoint)
            cc = cv_distance(CentralPoint,DefaultTopPoint)
            RotationAngle =  math.degrees(math.acos((aa*aa-bb*bb-cc*cc)/(-2*bb*cc)))#旋转角
            if Sdirection < 0: RotationAngle = 360-RotationAngle
        # print("RotationAngle:",RotationAngle)
    return realPosition,RotationAngle
def cv_distance(a,b):
    #求出两点的截距
   selfx=a[0]-b[0]
   selfy=a[1]-b[1]
   selflen= math.sqrt((selfx**2)+(selfy**2))
   return selflen




def run():
    print("start Reveive")
    cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.115//stream1')
    while True:
        ret, frame = cap.read()
        # frame=cv2.imread("/home/mr/cam_file/recognition_QRcode/opencv_qr/b.png")
        realPosition,RotationAngle = conter(frame)
        cv2.imshow('123456789',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':

    run()
    pass

