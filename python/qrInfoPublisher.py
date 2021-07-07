# -*- coding:utf-8 -*-

import time
import multiprocessing as mp
import numpy as np
import sys
import rospy
# from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import cv2
import pyzbar.pyzbar as pyzbar
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def talker(x,y,rikirobot,cutImage):
    bridge = CvBridge()
    #imagemsg = bridge.cv2_to_imgmsg(cutImage, encoding="bgr8")
    
    array = [x,y,rikirobot]
    qrCodePubArray = Float64MultiArray(data = array)
    rospy.loginfo(qrCodePubArray)
    pub.publish(qrCodePubArray)
    #pub2.publish(imagemsg)
    rate.sleep()
def decodeDisplay(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (y, x, w, h) = barcode.rect  #1080*1920
        cutImage = image[x-50:x+w+50,y-50:y+h+50]    #对图像进行剪裁
        #cv2.imshow('thresh1',cutImage)
        # print(barcode.rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image, (y, x), (y + h, x + w), (0, 0, 255), 2)
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # position_cal(x+w/2,y+h/2,barcodeData)

        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (y, x - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)
 
        # 向终端打印条形码数据和条形码类型

        rikirobot = float(barcodeData[-1:])
        talker(x,y,rikirobot,cutImage)
        # print("#####################")
    # return image

def run():
    print("start Reveive")
    cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.115//stream1')
    # print(frame.shape)
    # plt.imshow(frame,'gray')
    # plt.show()
    while True:
        ret, frame = cap.read()
        decodeDisplay(frame)
        cv2.imshow('123456789',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    # run_single_camera()
    rospy.init_node('qrCodePubArray',anonymous=True)
    pub = rospy.Publisher('qrCodePubArray',Float64MultiArray, queue_size=10)
    pub2 = rospy.Publisher('qrCodePubImage',Image, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    run()
    pass
