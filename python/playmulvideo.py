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
def image_put(q, user, pwd, ip):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//stream1" % (user, pwd, ip))
    if cap.isOpened():
        print('%sready'%ip)

    while True:
        #q.put(cap.read()[1])
        q.put(cv2.resize(cap.read()[1],(IMAGE_WIDTH,IMAGE_HEIGHT)))
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(window_name, IMAGE_WIDTH, IMAGE_HEIGHT)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_multi_camera():
    # user_name, user_pwd = "admin", "password"
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

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
def image_collect(queue_list, camera_ip_l):
    import numpy as np

    """show in single opencv-imshow window"""
    window_name = "%s_and_so_no" % camera_ip_l[0]
    cv2.resizeWindow(window_name, IMAGE_WIDTH*2, IMAGE_HEIGHT*2)
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        imgs = [q.get() for q in queue_list]

        image1 = np.concatenate([imgs[2],imgs[1],imgs[0]], axis=0)
        image2 = np.concatenate([imgs[5],imgs[4],imgs[3]], axis=0)
        image3 = np.concatenate([imgs[8],imgs[7],imgs[6]], axis=0)
        imgs = np.concatenate([image1,image2,image3],axis=1)
        print(imgs.shape)
        cv2.resizeWindow(window_name, IMAGE_WIDTH*3, IMAGE_HEIGHT*3)
        cv2.imshow(window_name, imgs)
        cv2.waitKey(1)
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

    processes = [mp.Process(target=image_collect, args=(queues, camera_ip_l))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()

if __name__ == '__main__':
    #run_multi_camera()
    run_multi_camera_in_a_window()  # with 1 + n threads
    pass
