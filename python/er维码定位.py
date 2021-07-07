# encoding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cv2
import pyzbar.pyzbar as pyzbar
from matplotlib import pyplot as plt
from time import time
import math

def decodeDisplay(image):
    #取出二维码区域并转为灰度图，求出三个定位点，返回三个定位点的坐标已经二维码区域的中心点坐标
    lista = []
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # (x, y, w, h) = barcode.rect
        (y, x, w, h) = barcode.rect#暂时不知道什么缘故xy需要互换位置
        print("x:{} y:{} w:{} h:{}".format(x, y,w,h))
        centerRotation = [w/2,h/2]
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        thresh1 = image[x:x+h,y:y+w]    #对数组进行剪裁
        plt.imshow(thresh1,'gray')
        plt.show()
        GrayImage=cv2.cvtColor(thresh1,cv2.COLOR_BGR2GRAY)
        ret,frame=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
        # ret,frame=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY_INV)
        plt.imshow(frame,'gray')
        plt.show()
        start = time()
        for pix_x in range(5,h-5):
            for pix_y in range(5,w-5):
                # gery = image[pix_x][pix_y]
                # print("pix_x:{} pix_y:{} gery:{}".format(pix_x, pix_y, gery))
                # if gery==0:
                #     nb +=1
                nb = 0
                for row in range(pix_x-5,pix_x+5):
                    for clo in range(pix_y-5,pix_y+5):
                        gery = frame[row][clo]
                        # print("row:{} clo:{} gery:{}".format(row, clo, gery))
                        if gery==0:
                            nb +=1
                # print(nb)
                if(nb>99):
                    lista.append([pix_y,pix_x])
        stop = time()
        print(str(stop-start) + "秒")
    print(lista)
    return image,lista,centerRotation
def intercept(a,b):
    #求出两点的截距
   selfx=a[0]-b[0]
   selfy=a[1]-b[1]
   selflen= math.sqrt((selfx**2)+(selfy**2))
   return selflen
def SeekingRightAngles(lista):
    # 求三点连成三角形的三个角度，返回最大角度（直角）所对应的点
   a = intercept(lista[0],lista[1]) #intercept01;
   b = intercept(lista[1],lista[2]) #intercept12;
   c = intercept(lista[2],lista[0]) #intercept20;
   A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))#点2处的夹角
   B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))#点0处的夹角
   C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))#点1处的夹角
   print(A)
   print(B)
   print(C)
   if A>B and A>C:return lista[2]
   if B>A and B>C:return lista[0]
   if C>B and C>A:return lista[1]
def RotationAngle(rightpoint,centerRotation):
    # 以左上角45度为参照求旋转角度，旋转角度为二维码区域的中心点分别与 （0，0） 以及 90度角对应的点 的 连线 的夹角
   lista = [[0,0],rightpoint,centerRotation]
   slope = rightpoint[1]/rightpoint[0]#求直角点的斜率，用于判断顺时针还是逆时针旋转
   if slope == 1:#如果斜率等于1 则有0和180度两种可能
      if rightpoint[0]<centerRotation[0]/2: return 0#关键点的x坐标小于旋转中心的x坐标
      else:return 180
   else: 
      a = intercept(lista[0],lista[1]) #intercept01;
      b = intercept(lista[1],lista[2]) #intercept12;
      c = intercept(lista[2],lista[0]) #intercept20;
      A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))#旋转角
      if slope<1: return -A#如果斜率小于1 逆时针旋转
      if slope>1: return A#如果斜率大于1 顺时针旋转
i = 0
# while True:
i = i+1
print('##########################')
print(i)
print('##########################')
# frame=cv2.imread('/home/lzq/file/rqcode/2020.png')   

# print(frame.shape)#1080*1920
# print(frame[369][712])

print("start Reveive")
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.115//stream1')
ret, frame = cap.read()
print(frame.shape)
plt.imshow(frame,'gray')
plt.show()
frame,lista,centerRotation = decodeDisplay(frame)

# rightpoint = SeekingRightAngles(lista)
# print(rightpoint)
# rotationAngle = RotationAngle(rightpoint,centerRotation)
# print(rotationAngle)

# plt.imshow(frame,'gray')
# plt.show()

