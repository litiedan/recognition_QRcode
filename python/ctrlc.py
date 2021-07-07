import multiprocessing
import time
import signal
import sys

jobs = []

def worker():
    signal.signal(signal.SIGINT, signal_handler)
    while(True):
        time.sleep(1.1234)
        print("Working...")

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    for p in jobs:
        p.terminate()
    sys.exit(0)

if __name__ == "__main__":
    for i in range(50):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()
