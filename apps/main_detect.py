from collections import defaultdict, deque
from multiprocessing import Process, Queue
from datetime import datetime
import utils_yolo
import os
import sys
import time
from threading import Thread, RLock
from ultralytics import YOLO
import numpy as np
import cv2
import itertools
from PIL import Image
import queue as queue_lib
import time

def annotate_img_opencv(image, annotations):
    """
    Annotate the image with bounding boxes and labels.

    Args:
    - image: OpenCV image (numpy array)
    - annotations: List of dictionaries containing annotation bounds and tag names

    Returns:
    - Annotated OpenCV image (numpy array)
    """
    # Iterate through each annotation
    for annotation in annotations:
        bounds = annotation["bounds"]
        tag_name = annotation.get("tagName")
        confidence = round(float(annotation.get("confidence")), 2)
        text= str(tag_name)+" - "+str(confidence)
        
        x1 = int(bounds["x1"])
        y1 = int(bounds["y1"])
        x2 = int(bounds["x2"])
        y2 = int(bounds["y2"])
        x3 = int(bounds["x3"])
        y3 = int(bounds["y3"])
        x4 = int(bounds["x4"])
        y4 = int(bounds["y4"])
        
        # Define the points of the polygon
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Draw the polygon on the image
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # Get the width and height of the text box
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)        
        
        # Calculate the position for the text box
        text_x = x1
        text_y = y1 - 10  # slightly above the top-left corner of the bounding box
        
        # Ensure the text box is within the image bounds
        text_x = max(0, text_x)
        text_y = max(text_height + baseline, text_y)
        
        # Draw a filled rectangle for the text background
        cv2.rectangle(image, (text_x, text_y - text_height - baseline), 
                      (text_x + text_width, text_y + baseline), 
                      (0, 0, 255), thickness=cv2.FILLED)
        
        # Put the text on the image
        cv2.putText(image, text, (text_x, text_y - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
    
    return image

def annotate_img_opencv_old(image, annotations):
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
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # Codec for the video stream
    # out = cv2.VideoWriter(rtsp_url, fourcc, fps, (width, height))
    
    # Create VideoWriter object
        # ' !videorate max-rate=4 ' + \
    out = cv2.VideoWriter('appsrc ! videoconvert' + \
        # ' !videorate max-rate=4 ' + \
        ' ! video/x-raw,format=I420' + \
        ' ! x264enc  speed-preset=medium tune=zerolatency bitrate=2000' \
        ' ! video/x-h264,profile=baseline' + \
        ' ! rtspclientsink location=' + rtsp_url,
        cv2.CAP_GSTREAMER, fourcc, fps, (width, height), True)
    if not out.isOpened():
        raise RuntimeError("Can't open video writer")
    return out

def create_writer_hls(hls_url,width,height,fps):
     # Define VideoWriter properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video stream
    # out = cv2.VideoWriter(rtsp_url, fourcc, fps, (width, height))
    
    # Create VideoWriter object
        # ' !videorate max-rate=4 ' + \
    out = cv2.VideoWriter('appsrc ! videoconvert' + \
        # ' !videorate max-rate=4 ' + \
        ' ! video/x-raw,format=I420' + \
        ' ! x264enc  speed-preset=slow tune=zerolatency bitrate=5000' \
        ' ! mpegtsmux' + \
        ' ! hlssink location=' + hls_url,
        cv2.CAP_GSTREAMER, fourcc, fps, (width, height), True)
    if not out.isOpened():
        raise RuntimeError("Can't open video writer")
    return out

def read_frames(stream_url, aggr_queue, queues, to_print=False):
    try:
        if to_print:
            print("read_frames is printing")

        cap = None
        while cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("Unable to open stream, retrying in 1 second...")
                time.sleep(1)

        frames = 0
        while True:
            start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()  # Release the current capture before reinitializing
                if not cap.isOpened():
                    print("Unable to open stream, retrying in 2 seconds...")
                    time.sleep(2)
                cap = cv2.VideoCapture(stream_url)
                continue

            frames += 1
            aggr_queue.put(frame)
            if frames % 300 == 0:
                print(f"Reading... - frame {frames}")
            for queue in queues:
                queue.put(frame)

            if to_print:
                elapsed = time.time() - start
                print("fps_so_far:", 1 / elapsed)

    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    finally:
        if cap is not None:
            cap.release()
    print("read_frames completed")

def read_frames_old(stream_url, aggr_queue, queues, to_print=False):
    try:
        if to_print:
            print("read_frames is printing")
        cap = cv2.VideoCapture(stream_url)
        frames = 0
        while True:
            
            start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Waiting for stream...")
                time.sleep(1)
                continue
            frames = frames + 1
            aggr_queue.put(frame)
            if frames % 50 == 0:
                print("Reading")
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

def save_image_with_timestamp(image):
    # Generate a timestamp
    image_timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")

    # Check if the path /shared/test_stream/ exists, create it if it doesn't
    directory_path = "/shared/test_stream/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Create the full file path
    file_path = os.path.join(directory_path, f"image_{image_timestamp}.png")

    # Save the image using OpenCV
    cv2.imwrite(file_path, image)
    print(f"Image saved at: {file_path}")

def consumer(queue_in, to_print=False, confidence_filters=None):
    try:
        rtsp_url = "rtsp://localhost:8554/mystream"
        hls_url = "rtsp://localhost:8888/mystream"
        fps = 30
        out = None
        out_w = None
        out_h = None
        if to_print:
            print("consumer is printing")
        frames = 0
        init_frame = 50

        # TODO REMOVE
        # out_w, out_h = 300, 300
        # out = create_writer(rtsp_url, out_w, out_h, fps)
        last_loop = time.time()
        #print("Streaming...")

        while True:
            processed_frame = queue_in.get()
            frame = processed_frame["frame"]
            res_dicts = processed_frame["res_dicts"]
            frame_h, frame_w = frame.shape[:2]
            if out is None or frame_h != out_h or frame_w != out_w:
               out_h, out_w = frame_h, frame_w
               out = create_writer(rtsp_url, frame_w, frame_h, fps)
               print("Output created")
               #out_hls = create_writer_hls(hls_url, frame_w, frame_h, fps)
            frames=frames+1
            if frames >= init_frame:
                filtered_objects = utils_yolo.get_filtered_objects(res_dicts, confidence_filters)
                annotated_frame = annotate_img_opencv(frame, filtered_objects)
                
                out.write(annotated_frame)
                #out_hls.write(annotated_frame)
                if frames % 300 == 0 :  
                    ts = time.time()


    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("consumer", "completed")

if __name__ == "__main__":
    """
    Usage: demo_multiprocess.py stream_url 
    """

    n_producers = 1

    queue_read2aggr = Queue()
    queues_read2pred = [Queue() for _ in range(n_producers)]
    queues_pred2aggr = [Queue() for _ in range(n_producers)]
    queue_aggr2cons = Queue()

    queues = [queue_read2aggr, queue_aggr2cons] + queues_read2pred + queues_pred2aggr
    processes = []

    stream_url = sys.argv[1]
    read_frames_process = Process(target=read_frames, args=(stream_url, queue_read2aggr, queues_read2pred))
    processes.append(read_frames_process)

    yolo_paths = [
        "../apps/models/yolo_VDTMLT_1024p.pt",
        "../apps/models/best_tanks_militaryTrucks.pt",
    ]
    yolo_confidence_filters = [
        dict(pedestrian=0.5, car=0.65, van=0.5, truck=0.3, military_tank=0.6, military_truck=0.4, military_vehicle=0.4),
        dict(military_tank=0.05, military_truck=0.05),
    ]
    
    predictor_procesesses = [
        Process(
            target=predictor,
            args=(yolo_paths[i], queues_read2pred[i], queues_pred2aggr[i], i)
        ) for i in range(n_producers)
    ]
    processes += predictor_procesesses

    aggregator_process = Process(target=aggregator,
        args=(queue_read2aggr, queues_pred2aggr, queue_aggr2cons))
    processes.append(aggregator_process)

    consumer_process = Process(target=consumer, 
        args=(queue_aggr2cons, False, yolo_confidence_filters))
    processes.append(consumer_process)

    for process in processes:
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("main", "completed")


    # # Check if there were any exceptions
    # if not error_queue.empty():
    #     #print("An error occurred in the consumer process:")
    #     with open('logs_ww.txt', 'a') as f:
    #         f.write("An error occurred in the consumer process:\n")
    #         f.write(error_queue.get())
    #         f.write("\n")  # Ensure a new line after the traceback