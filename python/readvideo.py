#coding=utf-8
#首先导入本次所需要的库，最后一个csv是Python自带的csv表格操作库，这里我们需要把我们扫到的二维码信息都存入csv表格里。
import cv2
import Queue
from pyzbar import pyzbar
import csv
import threading
q=Queue.Queue()
#然后我们设置一个变量，来存放我们扫到的二维码信息，我们每次扫描一遍都会要检测扫描到的二维码是不是之前扫描到的，
# 如果没有就存放到这里。接着我们调用opencv的方法来实例化一个摄像头，
# 最后我们设置一些我们存放二维码信息的表格的路径。
found = set()

#存放数据的表格
PATH = "test.csv"

def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.112//stream1')
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)
 
 
def Display():
     print("Start Displaying")
     while True:
        if q.empty() !=True:
            frame=q.get()
            cv2.imshow("frame1", frame)

            test = pyzbar.decode(frame)
    
            # 循环检测到的条形码
            for tests in test:
                # 先将它转换成字符串
                testdate = tests.data.decode('utf-8')
                testtype = tests.type
        
                # 绘出图像上条形码的数据和条形码类型
                printout = "{} ({})".format(testdate, testtype)
        
                if testdate not in found:
                # 向终端打印条形码数据和条形码类型
                    print("[INFO] Found {} barcode: {}".format(testtype, testdate))
                    print(printout)
                #存放扫描数据
                if testdate not in found:
                    with open(PATH,'a+') as f:
                    #a+ 以附加方式打开可读写的文件。若文件不存在，则会建立该文件，如果文件存在，写入的数据会被加到文件尾后，即文件原先的内容会被保留。
                        csv_write = csv.writer(f)
                        date = [testdate]
                        csv_write.writerow(date)
                    found.add(testdate)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__=='__main__':
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
