# -*- coding:utf-8 -*-

import time
import multiprocessing as mp
import numpy as np
import sys
import rospy
# from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
import cv2
import pyzbar.pyzbar as pyzbar

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
## '地图长度'
x_length = 4.8
## '地图宽度'
y_width = 3.6

x_pixel = IMAGE_WIDTH * 2
y_pixel = IMAGE_HEIGHT * 2

# pts_o1 = np.float32([[216, 43], [743.5, 16], [235.8, 437.9], [763.4, 413.8]])
# pts_o2 = np.float32([[158.5, 90.5], [682.5, 83.4], [160.6, 488.3], [691.8, 477]])
# pts_o3 = np.float32([[196.8, 85.5], [721.5, 97.6], [185.5, 486.2], [718.7, 492.6]])

# pts_o4 = np.float32([[242.9, 66.4], [769.8, 47.2], [252.1, 460.6], [788.9, 452.8]])
# pts_o5 = np.float32([[209.6, 84.8], [734.3, 96.9], [198.2, 481.9], [728.6, 491.8]])
# pts_o6 = np.float32([[206, 92.6], [727.2, 106.1], [191.8, 489.7], [727.9, 501.8]])

# pts_o7 = np.float32([[220.2, 67.1], [745.7, 50.1], [221.6, 467.7], [759.8, 451.4]])
# pts_o8 = np.float32([[174.8, 58.6], [706.7, 69.2], [172, 457.8], [697.4, 464.2]])
# pts_o9 = np.float32([[147.9, 59.3], [683.3, 49.4], [156.4, 459.2], [683.3, 445.8]])

# 标定
pts_o1 = np.float32([[202.5, 33.1], [760.5, 1.9], [225.2, 445], [918, 414.5]])
pts_o2 = np.float32([[136.5, 78.4], [688.2, 77.7], [135.8, 506.7], [701.7, 486.9]])
pts_o3 = np.float32([[184, 75.6], [734.3, 90.5], [167.7, 499.6], [734.3, 506]])

pts_o4 = np.float32([[230.1, 57.2], [787.5, 32.3], [240.8, 471.3], [805.2, 460.6]])
pts_o5 = np.float32([[198.9, 77], [747.1, 89.1], [179.1, 496.1], [745.7, 507.4]])
pts_o6 = np.float32([[192.6, 81.3], [738.6, 98.3], [172, 508.2], [741.4, 517.4]])

pts_o7 = np.float32([[206.7, 55.7], [757.7, 37.3], [211.6, 472.7], [776.9, 458.5]])
pts_o8 = np.float32([[153.6, 41.6], [716.7, 60.7], [150, 473.4], [706.7, 475.5]])
pts_o9 = np.float32([[121.6, 42.3], [691.8, 40.9], [132.3, 477], [693.2, 455]])

# pts_o = [pts_o1,pts_o2,pts_o3,pts_o4,pts_o5,pts_o6,pts_o7,pts_o8,pts_o9]
pts_o = [pts_o5,pts_o6,pts_o8,pts_o9]
pts_d = np.float32([[0, 0], [960, 0], [0, 540], [960, 540]])

# jobs = []

# def signal_handler(signal):
    

M = []
for i in pts_o:
    M.append(cv2.getPerspectiveTransform(i, pts_d))

pub = rospy.Publisher('chatter', Point, queue_size=10)
rospy.init_node('rikirobot', anonymous=True)
rate = rospy.Rate(10) # 10hz

def talker(x,y,rikirobot):
    point = Point()
    rikirobot = rikirobot[-1:]
    point.x = y
    point.y = x
    point.z = float(rikirobot)
    
    # hello_str = "name:%s---x:%.2f---y:%.2f" % (rikirobot,x,y)
    rospy.loginfo(point)
    pub.publish(point)
    rate.sleep()


"""给定目标位置的像素计算实际位置"""
def position_cal(x, y,rikirobot):
    x_position = (x_length / x_pixel) * x
    y_position = (y_width / y_pixel) * y
    print("name:%s---x:%.2f---y:%.2f" % (rikirobot,x_position,y_position))
    talker(x_position,y_position,rikirobot)
    # return x_position,y_position

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
 

def image_put(q, name, pwd, ip):
    
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//stream1" % (name, pwd, ip))
    # cap.set(cv2.CAP_PROP_FPS ,2)
    # cap = cap.resize()
    if cap.isOpened():
        print(ip)
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip))
        print('DaHua')
    
    # pro = True
    while cap.isOpened():
        # if pro:
        # q.put(cap.read()[1].resize((640,480)))
        # q.put(undistort(cv2.cvtColor(cv2.resize(cap.read()[1],(IMAGE_WIDTH,IMAGE_HEIGHT)),cv2.COLOR_BGR2GRAY)))
        # q.put(cv2.cvtColor(cv2.resize(cap.read()[1],(IMAGE_WIDTH,IMAGE_HEIGHT)),cv2.COLOR_BGR2GRAY))
        q.put(undistort(cv2.resize(cap.read()[1],(IMAGE_WIDTH,IMAGE_HEIGHT))))
        q.get() if q.qsize() > 1 else time.sleep(0.01)
            # pro = not pro
        
def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(0.2)

# zbar detect
def decodeDisplay(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        print(barcode.rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        position_cal(x+w/2,y+h/2,barcodeData)
 
        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)
 
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    # return image


def nine_to_one(queues):
    cv2.namedWindow('123456789', flags=cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('123456789', IMAGE_WIDTH*2, IMAGE_HEIGHT*2)
    while True:
        images = []
        for i in range(4):
            images.append(cv2.warpPerspective(queues[i].get(), M[i], (960, 540)))
        # image1 = np.concatenate([images[2],images[1],images[0]], axis=0)
        image2 = np.concatenate([images[1],images[0]], axis=0)
        image3 = np.concatenate([images[3],images[2]], axis=0)
        image = np.concatenate([image2,image3],axis=1)
        # image = nine_to_one(queues)
        # print(image.shape)
        decodeDisplay(image)

        cv2.imshow('123456789',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # return image

def run_multi_camera():
    # user_name, user_pwd = "admin", "password"
    user_name, user_pwd = "admin", "admin"
    camera_ip_l = [
        # '192.168.1.111',  # ipv4
        # '192.168.1.112',
        # '192.168.1.113',
        # '192.168.1.114',
        '192.168.1.115',
        '192.168.1.116',
        # '192.168.1.117',
        '192.168.1.118',
        '192.168.1.119',   
    ]

    mp.set_start_method(method='spawn',force=True,)  # init
    queues = [mp.Queue() for _ in camera_ip_l]

    processes = [mp.Process(target=nine_to_one, args=(queues,))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        # processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    # print('------------------')
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()

    # print('!!!')

    # cv2.namedWindow('123456789', flags=cv2.WINDOW_FREERATIO)
    # cv2.resizeWindow('123456789', 2880, 1620)
    # while True:
    #     # print('123')
    #     image = nine_to_one(queues)
    #     # print(image.shape)
    #     decodeDisplay(image)

    #     cv2.imshow('123456789',image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # print('You pressed Ctrl+C!')
    # for p in processes:
    #     p.terminate()
    # sys.exit(0)	

if __name__ == '__main__':
    # run_single_camera()
    run_multi_camera()
    pass
