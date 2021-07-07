# -*- coding:utf-8 -*-

import time
import multiprocessing as mp
import numpy as np
import sys
import rospy
# from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
import cv2
import pyzbar.pyzbar as pyzbar
from conterdetection import Conter
from conterdetection import conter
from datetime import datetime
import time
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

## '地图宽度'
map_width = 7.2
## '地图长度'
map_high = 5.4

frame_count = 0.001

qrcode_count = 0.001


# 标定
pts_o1 = np.float32([[275.5, 28.5], [838.5, 30.9], [263.2, 439], [822, 465.5]])
pts_o2 = np.float32([[170.4, 57.9], [723, 62.8], [165.5, 479.1], [725.8, 477.9]])
pts_o3 = np.float32([[194.7, 68.6], [746.3, 81.5], [179.7, 490.6], [744.3, 498]])

pts_o4 = np.float32([[223.1, 57.2], [780.5, 32.3], [233.8, 471.3], [794.2, 460.6]])
pts_o5 = np.float32([[206, 74], [752.1, 87.1], [191.8, 488], [750.7, 502.4]])
pts_o6 = np.float32([[194, 79], [739, 94], [176, 500.2], [743, 509]])

pts_o7 = np.float32([[206.7, 57.7], [757.7, 39.3], [211.6, 472.7], [776.9, 458.5]])
pts_o8 = np.float32([[153.6, 44.6], [716.7, 61.7], [150, 471.4], [706.7, 473.5]])
pts_o9 = np.float32([[125.6, 42.3], [685.8, 44.9], [137.3, 471], [690.2, 450]])

pts_o = [pts_o1,pts_o2,pts_o3,pts_o4,pts_o5,pts_o6,pts_o7,pts_o8,pts_o9]

pts_d = np.float32([[0, 0], [960, 0], [0, 540], [960, 540]])

    

M = []
for i in pts_o:
    M.append(cv2.getPerspectiveTransform(i, pts_d))


pub = rospy.Publisher('chatter', Point, queue_size=1)
pub3 = rospy.Publisher('angle3', Float32, queue_size=1)
pub5 = rospy.Publisher('angle5', Float32, queue_size=1)
pub6 = rospy.Publisher('angle6', Float32, queue_size=1)
rospy.init_node('rikirobot', anonymous=True)
rate = rospy.Rate(10) # 10hz

def talker(x,y,rikirobot,angle):
    point = Point()
    angle3_msg = Float32()
    angle5_msg = Float32()
    angle6_msg = Float32()

    rikirobot = rikirobot[-1:]
    point.x = x
    point.y = y
    point.z = float(rikirobot)

    rospy.loginfo(point)
    pub.publish(point)

    if point.z == 3:
        angle3_msg.data = angle
        rospy.loginfo(angle3_msg)
        pub3.publish(angle3_msg)
    elif point.z == 5:
        angle5_msg.data = angle
        rospy.loginfo(angle5_msg)
        pub5.publish(angle5_msg)
    elif point.z == 6:
        angle6_msg.data = angle
        rospy.loginfo(angle6_msg)
        pub6.publish(angle6_msg)
    rate.sleep()

def undistort(frame):
    fx = 773.420202580123
    cx = 472.795199168578
    fy = 767.901401454283
    cy = 259.419312238802
    k1, k2, p1, p2, k3 = -0.278286658240873, 0.076263277939532, 0.0, 0.0, 0.0
 
    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
 

def image_put(q, user, pwd, ip):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//stream1" % (user, pwd, ip))
    if cap.isOpened():
        print('%s is ready'%ip)

    while True:
        #q.put(cap.read()[1])
        q.put(undistort(cv2.resize(cap.read()[1],(IMAGE_WIDTH,IMAGE_HEIGHT))))
        #q.put(cv2.resize(cap.read()[1],(IMAGE_WIDTH,IMAGE_HEIGHT)))
        q.get() if q.qsize() > 1 else time.sleep(0.01)        
def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(0.2)

# zbar detect
def decodeDisplay(image):
    global frame_count
    global qrcode_count
    frame_count = frame_count+1
    #print("frame_count: ",frame_count)
    
#    print(time_a)
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        #print(barcode.rect)
        cutImage = image[y-30:y+h+30,x-30:x+w+30]
        cv2.imshow('thresh1',cutImage)
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        #realPosition,RotationAngle = conter(image)
        realPosition,RotationAngle = Conter(x,y,cutImage,IMAGE_WIDTH*3, IMAGE_HEIGHT*3,map_width,map_high)
        if realPosition[0] > -1:
            talker(realPosition[0],realPosition[1],barcodeData,RotationAngle)
            qrcode_count = qrcode_count+1
            #print("qrcode_count: ",qrcode_count)
        # 绘出图像上条形码的数据和条形码类型
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)

    
    #print("二维码识别成功率：",qrcode_count/frame_count)
    print("#######################################")
def nine_to_one(queues):
    #cv2.namedWindow('123456789')
    #cv2.resizeWindow('123456789', 1920, 1080)
    while True:
        time_a = datetime.now() #获得当前时间
        images = []
        for i in range(9):
            images.append(cv2.warpPerspective(queues[i].get(), M[i], (IMAGE_WIDTH, IMAGE_HEIGHT)))
            #images.append(queues[i].get())
        # image1 = np.concatenate([images[2],images[1],images[0]], axis=0)
        #print(images[0].shape)
        #image = images[0]
        image1 = np.concatenate([images[2],images[1],images[0]], axis=0)
        image2 = np.concatenate([images[5],images[4],images[3]], axis=0)
        image3 = np.concatenate([images[8],images[7],images[6]], axis=0)
        image = np.concatenate([image1,image2,image3],axis=1)
        # image = nine_to_one(queues)
        #print(image.shape)
        decodeDisplay(image)
        time_b = datetime.now()  # 获取当前时间
        durn = (time_b-time_a).microseconds/1000#两个时间差，并以毫秒显示出来
        print("处理一帧的时长：",durn)

        image = cv2.resize(image,(1920,1080))
        window_name = 'cam'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name,image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # return image

def run_multi_camera_in_a_window():
    user_name, user_pwd = "admin", "admin"
    camera_ip_l = [
         '192.168.31.111',  # ipv4
         '192.168.31.112',
         '192.168.31.113',
         '192.168.31.114',
         '192.168.31.115',
         '192.168.31.116',
         '192.168.31.117',
         '192.168.31.118',
         '192.168.31.119',   
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=9) for _ in camera_ip_l]

    processes = [mp.Process(target=nine_to_one, args=(queues,))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()
if __name__ == '__main__':
    run_multi_camera_in_a_window()
    pass
