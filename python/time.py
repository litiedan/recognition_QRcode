from datetime import datetime
import time

time_a = datetime.now() #获得当前时间
print(time_a)
time.sleep(1)      #睡眠两秒
time_b = datetime.now()  # 获取当前时间
print(time_b)
durn = (time_b-time_a).microseconds#两个时间差，并以毫秒显示出来
print(durn)

timeshow = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))  #获取当前时间 ，并以当前格式显示
print(timeshow)

