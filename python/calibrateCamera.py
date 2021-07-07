import cv2
import numpy as np
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.111//stream1')


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
 
while(cap.isOpened()):
    ret, frame = cap.read()
   # frame =
    cv2.imshow('frame', undistort(cv2.resize(frame,(960,540))))
    #cv2.imshow('frame',frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

