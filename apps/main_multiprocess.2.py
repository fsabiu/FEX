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
import queue as queue_lib

def annotate_img_opencv(image, annotations):
    """
    Annotate the image with bounding boxes.

    Args:
    - image: OpenCV image (numpy array)
    - annotations: List of dictionaries containing annotation bounds

    Returns:
    - Annotated OpenCV image (numpy array)
    """
    # Iterate through each annotation
    for annotation in annotations:
        bounds = annotation["bounds"]
        x1 = bounds["x1"]
        y1 = bounds["y1"]
        x2 = bounds["x2"]
        y2 = bounds["y2"]
        x3 = bounds["x3"]
        y3 = bounds["y3"]
        x4 = bounds["x4"]
        y4 = bounds["y4"]
        
        # Define the points of the polygon
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Draw the polygon on the image
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    
    return image

def create_writer(rtsp_url,width,height,fps):
     # Define VideoWriter properties
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for the video stream
    # out = cv2.VideoWriter(rtsp_url, fourcc, fps, (width, height))
    
    # Create VideoWriter object
        # ' !videorate max-rate=4 ' + \
    out = cv2.VideoWriter('appsrc ! videoconvert' + \
        ' ! video/x-raw,format=I420' + \
        ' ! x264enc  speed-preset=medium tune=zerolatency bitrate=800' \
        ' ! video/x-h264,profile=baseline' + \
        ' ! rtspclientsink location=' + rtsp_url,
        cv2.CAP_GSTREAMER, fourcc, fps, (width, height), True)
    if not out.isOpened():
        raise RuntimeError("Can't open video writer")
    return out

def read_frames(stream_url, aggr_queue, queues, to_print=False):
    try:
        if to_print:
            print("read_frames is printing")
        cap = cv2.VideoCapture(stream_url)
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("read_frames failure")

            aggr_queue.put(frame)
            for queue in queues:
                queue.put(frame)

            if to_print:
                elapsed = time.time() - start
                print("fps_so_far:", 1/elapsed)
    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("read_frames", "completed")

def predictor(model_path, queue_in, queue_out, p_id):
    try:
        model = YOLO(model_path)
        while True:
            frame = queue_in.get()
            results = model.predict(source=frame, device=f'cuda:{p_id}', verbose=False)

            res_dict = utils_yolo.get_result_dict(model, results[0])
            res_dict["yolo_producer"] = p_id

            queue_out.put({"yolo_producer": p_id, "res_dict": res_dict})
    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("predictor", p_id, "completed")

def aggregator(frame_queue, queues_in, queue_out, confidence_filters=None):
    try:
        while True:
            frame = frame_queue.get()
            # imgs_info = dict()
            res_dicts = dict()

            # per ora bloccante
            for i, queue_in in enumerate(queues_in):
                res_dicts[i] = queue_in.get()["res_dict"]


            queue_out.put({
                "frame": frame,
                "res_dicts": res_dicts
            })
    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("aggregator", "completed")

def consumer(queue_in, to_print=False, confidence_filters=None):
    try:
        rtsp_url = "rtsp://localhost:8554/mystream"
        fps = 30
        out = None
        out_w = None
        out_h = None
        if to_print:
            print("consumer is printing")
        frames = 0
        init_frame = 50

        # TODO REMOVE
        out_w, out_h = 300, 300
        out = create_writer(rtsp_url, out_w, out_h, fps)

        while True:
            processed_frame = queue_in.get()
            frame = processed_frame["frame"]
            frame_h, frame_w = frame.shape[:2]
            if out is None or frame_h != out_h or frame_w != out_w:
                out_h, out_w = frame_h, frame_w
                out = create_writer(rtsp_url, frame_w, frame_h, fps)
                
            filtered_objects = utils_yolo.get_filtered_objects(frame, confidence_filters)
            annotated_frame = annotate_img_opencv(frame, filtered_objects)
            out.write(annotated_frame)
            print(np.shape(annotated_frame))
            raise ValueError("OK")
            
            frames = frames + 1
            if frames==init_frame:
                start = time.time()
            elif frames<=init_frame:
                if to_print:
                    print("consumer:", frames)
            if to_print and frames >= init_frame:
                elapsed = time.time() - start
                print("consumer: ", (frames-init_frame)/elapsed)
    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("consumer", "completed")

if __name__ == "__main__":
    """
    Usage: demo_multiprocess.py stream_url 
    """

    # n_producers = 2

    # # yolo1_confidence_filters = {
    # #     "military_tank": 0.05,
    # #     "military_truck": 0.05
    # # }

    # # yolo2_confidence_filters = {
    # #     "military_tank": 0.05,
    # #     "military_truck": 0.05
    # # }

    # queue_read2aggr = Queue()
    # queues_read2pred = [Queue() for _ in range(n_producers)]
    # queues_pred2aggr = [Queue() for _ in range(n_producers)]
    # queue_aggr2cons = Queue()

    # queues = [queue_read2aggr, queue_aggr2cons] + queues_read2pred + queues_pred2aggr
    # processes = []

    # stream_url = sys.argv[1]
    # read_frames_process = Process(target=read_frames, args=(stream_url, queue_read2aggr, queues_read2pred))
    # processes.append(read_frames_process)

    # yolo_paths = [
    #     "../apps/models/best_tanks_militaryTrucks.pt",
    #     "../apps/models/best_tanks_militaryTrucks.pt",
    # ]
    # yolo_confidence_filters = [
    #     dict(military_tank=0.05, military_truck=0.05),
    #     dict(military_tank=0.05, military_truck=0.05),
    # ]
    
    # predictor_procesesses = [
    #     Process(
    #         target=predictor,
    #         args=(yolo_paths[i], queues_read2pred[i], queues_pred2aggr[i], i)
    #     ) for i in range(n_producers)
    # ]
    # processes += predictor_procesesses

    # aggregator_process = Process(target=aggregator,
    #     args=(queue_read2aggr, queues_pred2aggr, queue_aggr2cons))
    # processes.append(aggregator_process)

    # consumer_process = Process(target=consumer, 
    #     args=(queue_aggr2cons, False, yolo_confidence_filters))
    # processes.append(consumer_process)

    # for process in processes:
    #     process.start()

    # try:
    #     for process in processes:
    #         process.join()
    # except KeyboardInterrupt:
    #     pass
    # except RuntimeError as E:
    #     print(E)
    # print("main", "completed")


    # # # Check if there were any exceptions
    # # if not error_queue.empty():
    # #     #print("An error occurred in the consumer process:")
    # #     with open('logs_ww.txt', 'a') as f:
    # #         f.write("An error occurred in the consumer process:\n")
    # #         f.write(error_queue.get())
    # #         f.write("\n")  # Ensure a new line after the traceback
    create_writer("rtsp://localhost:8554/mystream",1280,720,30)