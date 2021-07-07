# -*- coding:utf-8 -*-
'''
 视频拼接
'''
import cv2
import numpy as np
 
cam1 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.111//stream1')
cam2 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.112//stream1')
cam3 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.113//stream1')
cam8 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.118//stream1')
 
# # 获取视频1的宽度
# ww = int(cam1.get(3))
# # 获取视频1的高度
# hh = int(cam1.get(4))
# print(ww, hh)#1920*1080

# print(cam1.get(cv2.CAP_PROP_FPS))## 12帧/秒
# print(cam8.get(cv2.CAP_PROP_FPS))
 
while True:
    # 读取视频
    (ok1, frame1) = cam1.read()
    (ok2, frame2) = cam2.read()
    (ok3, frame3) = cam3.read()
    (ok8, frame8) = cam8.read()
    if ok1 and ok2 and ok3 and ok8:
        # # 重置视频大小，使两视频大小一致
        # frame1 = cv2.resize(frame1, (ww, hh))
        # frame2 = cv2.resize(frame2, (ww, hh))
        # 在视频中添加文字
        cv2.putText(frame1, "cam1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
        cv2.putText(frame2, "cam2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
        cv2.putText(frame3, "cam3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
        cv2.putText(frame8, "cam8", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
        # 拼接处理
        image = np.concatenate([frame1, frame2, frame3], axis=1)  # axis=0时为垂直拼接；axis=1时为水平拼接
        # 视频展示
        cv2.namedWindow('camera', 0)    
        cv2.resizeWindow('camera', 1920, 1080)
        cv2.imshow("camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
 
cam1.release()
cam2.release()
cam3.release()
cam8.release()
cv2.destroyAllWindows()