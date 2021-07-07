#coding=utf-8
import cv2
import pyzbar.pyzbar as pyzbar
import time
import matplotlib.pyplot as plt
import numpy as np
# 打开系统摄像头(0号)
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.31.119//stream1')
#cap = cv2.VideoCapture(0)


# 设置帧画面宽度
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# 设置帧画面高度
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 设置亮度
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
pts_o1 = np.float32([[275.5, 28.5], [838.5, 30.9], [263.2, 439], [822, 465.5]])
pts_o2 = np.float32([[170.4, 57.9], [723, 62.8], [165.5, 479.1], [725.8, 477.9]])
pts_o3 = np.float32([[194.7, 68.6], [746.3, 81.5], [179.7, 490.6], [744.3, 498]])

pts_o4 = np.float32([[223.1, 57.2], [780.5, 32.3], [233.8, 471.3], [794.2, 460.6]])
pts_o5 = np.float32([[201, 74], [747.1, 87.1], [185.8, 488], [745.7, 502.4]])
pts_o6 = np.float32([[194, 79], [739, 94], [176, 500.2], [743, 509]])

pts_o7 = np.float32([[206.7, 57.7], [757.7, 39.3], [211.6, 472.7], [776.9, 458.5]])
pts_o8 = np.float32([[153.6, 44.6], [716.7, 61.7], [150, 471.4], [706.7, 473.5]])
pts_o9 = np.float32([[125.6, 42.3], [685.8, 44.9], [137.3, 471], [690.2, 450]])

pts_d = np.float32([[0, 0], [960, 0], [0, 540], [960, 540]])

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

def decodeDisplay(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        print(barcode.polygon)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
 
        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)
 
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return image

# while cap.isOpened():
#     success, frame = cap.read()
#     start = time.time()
#     frame = undistort(cv2.resize(frame,(960,540)))
#     # frame = decodeDisplay(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
#     # frame = cv2.resize(frame,(960,540))
#     frame = decodeDisplay(frame)
#     #dst = cv2.undistort(frame, mtx, dist, None, mtx)
#     end = time.time()
#     fps = 1 / (end - start)
#     cv2.putText(frame, "FPS:%.2f"%fps, (850, 10), cv2.FONT_HERSHEY_SIMPLEX,
#                     .5, (0, 0, 125), 2)
#     if not success or cv2.waitKey(1) & 0xFF == 27:  # Esc键
#         break
#     #print(type(success))  
#     #print(success)
#     #print(type(frame))  
#     #print(frame.shape)
#     cv2.imshow("Camera", frame)
    # plt.imshow(frame)
success, frame = cap.read()
M = cv2.getPerspectiveTransform(pts_o9, pts_d)
frame = undistort(cv2.resize(frame,(960,540)))
plt.imshow(frame)
plt.show()
dst = cv2.warpPerspective(frame, M, (960, 540))
# decodeDisplay(dst)
# cv2.imshow('img',frame)
#cv2.imshow('dst',dst)
#plt.imshow(frame)
# plt.imshow(undistort(dst))
plt.imshow(dst)
plt.show()
# cv2.waitKey(0)
cap.release()
# cv2.destroyAllWindows()

# 1. 标定，恢复
# 2. 拼图
# 3. 数字识别
# 4. 定位
