# 同一张图里可能有多个二维码 也可能没有二维码，是以列表的格式存储二维码，so列表可能为空，也可能有多个元素
#每个二维码大约一万个像素点，每个像素点遍历其邻域一百个点，时间复杂度极高
import cv2
import pyzbar.pyzbar as pyzbar
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.112//stream1')
global i 
i = 0
# cap = cv2.VideoCapture(0)
# def decodeDisplay(image):
    # barcodes = pyzbar.decode(image)
#     for barcode in barcodes:
#         # 提取条形码的边界框的位置
#         # 画出图像中条形码的边界框
#         (x, y, w, h) = barcode.rect
#         print("x:{} y:{} w:{} h:{}".format(x, y,w,h))
#         # print(barcode.polygon)
#         # print(type(barcode.polygon))
#         # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
def decodeDisplay(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        print("x:{} y:{} w:{} h:{}".format(x, y,w,h))
        # (r,g,b) = image[x][y]
        # print("r:{} g:{} b:{}".format(r, g, b))

        for pix_x in range(x,x+w):
            for pix_y in range(y,y+h):
                (r,g,b) = image[pix_x][pix_y]
                print("r:{} g:{} b:{}".format(r, g, b))

                nb = 0
                # for row in range(pix_x-5,pix_x+5):
                #     for clo in range(pix_y-5,pix_y+5):
                #         # (r,g,b) = image[row][clo]
                #         # print("r:{} g:{} b:{}".format(r, g, b))
                #         if g==0:
                #             nb +=1
                # # print(nb)
                # if(nb>20):
                #     print("x:{} y:{}".format(pix_x, pix_y))
                

    return image
while cap.isOpened():
    success, frame = cap.read()
    i = i + 1
    print(i)
    print('#############################################')
    frame = decodeDisplay(frame)
    if not success or cv2.waitKey(1) & 0xFF == 27:  # Esc键
        break
    #print(type(success))  
    #print(success)
    #print(type(frame))  
    #print(frame.shape)
    # cv2.imshow("Camera", frame)
