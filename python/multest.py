#coding=utf-8
import cv2
i = 0
#cap01 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.111//stream1')
#cap02 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.112//stream1')
#cap03 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.113//stream1')
#cap04 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.114//stream1')
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.31.112//stream1')
#cap06 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.116//stream1')
#cap07 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.117//stream1')
#cap08 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.118//stream1')
#cap09 = cv2.VideoCapture('rtsp://admin:admin@192.168.1.119//stream1')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('outVideo.mp4', fourcc, fps, size)
while True:
#    if i%10 == 0:
        #success01, frame01 = cap01.read()
        #success02, frame02 = cap02.read()
        #success03, frame03 = cap03.read()
        #success04, frame04 = cap04.read()
        success05, frame05 = cap.read()
        #success06, frame06 = cap06.read()
        #success07, frame07 = cap07.read()
        #success08, frame08 = cap08.read()
        #success09, frame09 = cap09.read()
        if not success05 or cv2.waitKey(1) & 0xFF == 27:  # Esc键
            break
        if not success05 or cv2.waitKey(1) & 0xFF == 27:  # Esc键
            break
        out.write(frame05)  
        cv2.imshow("Camera05", frame05)
        #cv2.imshow("Camera02", frame02)
        # cv2.imshow("Camera03", frame03)
        # cv2.imshow("Camera04", frame04)
        #cv2.imshow("Camera05", frame05)
        # cv2.imshow("Camera06", frame06)
        # cv2.imshow("Camera07", frame07)
        # cv2.imshow("Camera08", frame08)
        # cv2.imshow("Camera09", frame09)
    # i = i + 1
#cap01.release()
#cap02.release()
out.release()
#cap03.release()
#cap04.release()
cap.release()
#cap06.release()
#cap07.release()
#cap08.release()
#cap09.release()
cv2.destroyAllWindows()
