from collections import defaultdict, deque
from multiprocessing import Process, Queue
import utils_yolo
import sys
import time
from threading import Thread, RLock
from ultralytics import YOLO
import numpy as np
import cv2
import itertools
from PIL import Image

def read_frames(stream_url):
    to_print=True
    if to_print:
        print("read_frames is printing")
    cap = cv2.VideoCapture(stream_url)
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("read_frames failure")

        if to_print:
            elapsed = time.time() - start
            print("fps:", 1/elapsed)

if __name__ == "__main__":
    """
    Usage: demo_multiprocess.py stream_url 
    """
    processes = []

    stream_url = sys.argv[1]
    read_frames_process = Process(target=read_frames, args=(stream_url,))
    processes.append(read_frames_process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()
