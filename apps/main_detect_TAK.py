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

def is_object_in_polygons(bounds, polygons):
    """
    Check if the object identified by bounds is included in any of the polygons.

    Args:
    - bounds: Dictionary containing the coordinates of the bounding box (x1, y1, ..., x4, y4)
    - polygons: List of polygons, where each polygon is a list of (x, y) tuples

    Returns:
    - Boolean: True if the bounding box is entirely within any polygon, False otherwise
    """
    # Extract the coordinates of the bounding box
    points = np.array([
        [bounds["x1"], bounds["y1"]],
        [bounds["x2"], bounds["y2"]],
        [bounds["x3"], bounds["y3"]],
        [bounds["x4"], bounds["y4"]]
    ], np.int32)

    # Check each polygon
    for polygon in polygons:
        # Convert the polygon to a numpy array of type float32
        poly_points = np.array(polygon, dtype=np.float32)

        # Check if all points of the bounding box are inside the polygon
        inside = all(cv2.pointPolygonTest(poly_points, (float(pt[0]), float(pt[1])), False) >= 0 for pt in points)
        
        # If all points are inside the polygon, return True
        if inside:
            return True
    
    # If no polygons contain the entire bounding box, return False
    return False

def annotate_img_opencv(image, annotations, pixel_mask):
    """
    Annotate the image with bounding boxes and labels.

    Args:
    - image: OpenCV image (numpy array)
    - annotations: List of dictionaries containing annotation bounds and tag names

    Returns:
    - Annotated OpenCV image (numpy array)
    """
    bbox_color = [
        (0, 0, 255),
        #(255, 0, 0)
        (0, 0, 255)
    ]
    masks = [
        [
            [(50, 540), (100, 540), (100, 560), (50,560)],
            [(50, 470), (100, 470), (100, 490), (50,490)],
            [(55, 395), (100, 395), (100, 415), (55,415)],
            [(55, 325), (100, 325), (100, 345), (55,345)],
            [(55, 255), (100, 255), (100, 270), (55,270)],
            [(65, 215), (220, 215), (220, 235), (65,235)],
            [(65, 105), (175, 105), (175, 125), (65,125)],
            [(65, 75), (175, 75), (175, 95), (65,95)],
            [(460, 40), (515, 40), (515, 55), (460,55)],
            [(585, 40), (630, 40), (630, 55), (585,55)],
            [(700, 40), (735, 40), (735, 55), (700,55)],
            [(815, 40), (845, 40), (845, 55), (815,55)],
            [(920, 40), (965, 40), (965, 55), (920,55)],
            [(700, 80), (735, 80), (735, 95), (700,95)]
        ]
    ]
    # Iterate through each annotation
    for annotation in annotations:
        bounds = annotation["bounds"]
        ob_id = -1
        if "id" in annotation:
            ob_id = annotation["id"]
        tag_name = annotation.get("tagName")
        confidence = round(float(annotation.get("confidence")), 2)
        text= str(tag_name)+" - "+str(confidence) #+f" id:{ob_id}"
        
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
        
        if not is_object_in_polygons(bounds, masks[pixel_mask]):
            # Draw the polygon on the image
            cv2.polylines(image, [points], isClosed=True, color=bbox_color[annotation["model_id"]], thickness=2)
            
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
                        bbox_color[annotation["model_id"]], thickness=cv2.FILLED)
            
            # Put the text on the image
            
            cv2.putText(image, text, (text_x, text_y - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
    
    return image

def pixel_to_gps(lat_dron, lon_dron, h_dron, pitch, yaw, roll, f, img_width, img_height, x_pixel, y_pixel):
    # Earth parameters
    R_EARTH = 6371e3  # Radius of the Earth in meters
    
    # Convert degrees to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)
    
    # Convert pixels to camera coordinates
    Z_cam = h_dron
    X_cam = (x_pixel - img_width / 2) * Z_cam / f
    Y_cam = (y_pixel - img_height / 2) * Z_cam / f
    
    # Rotation matrices
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Transform coordinates from camera frame to global frame
    cam_coords = np.array([X_cam, Y_cam, Z_cam])
    global_coords = R @ cam_coords
    
    # Convert global coordinates to GPS deltas
    d_lat = global_coords[1] / R_EARTH * (180 / np.pi)
    d_lon = global_coords[0] / (R_EARTH * np.cos(np.radians(lat_dron))) * (180 / np.pi)
    
    # New GPS coordinates
    new_lat = lat_dron + d_lat
    new_lon = lon_dron + d_lon
    
    return new_lat, new_lon

def send_to_tak(drone_dict, filtered_objects):
    print("Sending to TAK")

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

def read_drone_info(): # TO DO
    drone_dict = {}
    drone_dict["lat_dron"] = 52.2296756
    drone_dict["lon_dron"] = 21.0122287
    drone_dict["h_dron"] = 100  # Drone altitude in meters
    drone_dict["pitch"] = -10  # Camera pitch in degrees
    drone_dict["yaw"] = 45  # Azimuth in degrees
    drone_dict["roll"] = 0  # Roll in degrees
    drone_dict["f"] = 35  # Camera focal length in mm
    drone_dict["img_width"] = 4000  # Image width in pixels
    drone_dict["img_height"] = 3000  # Image height in pixels
    drone_dict["x_pixel"] = 2000  # x-coordinate of the pixel
    drone_dict["y_pixel"] = 1500  # y-coordinate of the pixel

    return drone_dict

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
            # metadata read function 
            if not ret or frame is None:
                cap.release()  # Release the current capture before reinitializing
                if not cap.isOpened():
                    print("Unable to open stream, retrying in 1 seconds...")
                    time.sleep(1)
                cap = cv2.VideoCapture(stream_url)
                continue
            
            frames += 1
            # Read drone coordinates
            drone_dict = read_drone_info() # TO DO

            aggr_queue.put((frame, drone_dict))
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

def predictor(model_path, queue_in, queue_out, p_id):
    try:
        model = YOLO(model_path)
        while True:
            frame = queue_in.get()
            start = time.time()
            #results = model.track(source=frame, device=f'cuda:{p_id}', verbose=False, persist=True)
            results = model.predict(source=frame, device=f'cuda:{p_id}', verbose=False)
            end = time.time()
            res_dict = utils_yolo.get_result_dict(model, results[0])
            res_dict["yolo_producer"] = p_id
            print(f"elapsed {end-start}")





            queue_out.put({"yolo_producer": p_id, "res_dict": res_dict})
    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("predictor", p_id, "completed")

def aggregator(frame_queue, queues_in, queue_out, filter_confidence_filters=None):
    try:
        while True:
            frame, drone_dict = frame_queue.get()
            # imgs_info = dict()
            res_dicts = dict()

            # per ora bloccante
            for i, queue_in in enumerate(queues_in):
                res_dicts[i] = queue_in.get()["res_dict"]

            queue_out.put({
                "frame": frame,
                "res_dicts": res_dicts,
                "drone_dict": drone_dict
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

def consumer(queue_in, pixel_mask, to_print = False, confidence_filters = None):
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
        last_loop = time.time()

        n_hist = 10
        history = deque(maxlen=n_hist)
        next_id = 0

        while True:
            processed_frame = queue_in.get()
            frame = processed_frame["frame"]
            res_dicts = processed_frame["res_dicts"]
            drone_dict = processed_frame["drone_dict"]

            frame_h, frame_w = frame.shape[:2]
            if out is None or frame_h != out_h or frame_w != out_w:
               out_h, out_w = frame_h, frame_w
               out = create_writer(rtsp_url, frame_w, frame_h, fps)
               print("Output created")
               #out_hls = create_writer_hls(hls_url, frame_w, frame_h, fps)
            frames=frames+1
            if frames >= init_frame:
                # Filtering
                filtered_objects = utils_yolo.get_filtered_objects(res_dicts, confidence_filters)

                # Getting ids
                filtered_objects, history, next_id = assign_ids(filtered_objects, history, next_id)
                
                annotated_frame = annotate_img_opencv(frame, filtered_objects, pixel_mask)
                #print(current_frame_objects)

                # Sendiing
                send_to_tak(drone_dict, filtered_objects)
                out.write(annotated_frame)

                if frames % 300 == 0 :  
                    ts = time.time()


    except KeyboardInterrupt:
        pass
    except RuntimeError as E:
        print(E)
    print("consumer", "completed")

def initialize_history(history):
    history.clear()
    return history

def calculate_iou(box1, box2):
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x3'], box2['x3'])
    y2 = min(box1['y3'], box2['y3'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1['x3'] - box1['x1']) * (box1['y3'] - box1['y1'])
    box2_area = (box2['x3'] - box2['x1']) * (box2['y3'] - box2['y1'])

    union = box1_area + box2_area - intersection
    iou = intersection / union if union != 0 else 0
    return iou

def assign_ids(current_detections, history, next_id):

    # Flatten the history to a list of all previous objects with their IDs
    past_objects = [(obj, frame_id, obj_id) for frame_id, frame in enumerate(history) for obj_id, obj in frame.items()]

    # Prepare the current frame's objects dictionary
    current_frame_objects = {}

    for detection in current_detections:
        best_match = None
        best_iou = 0.0
        for past_obj, _, past_id in past_objects:
            iou = calculate_iou(detection['bounds'], past_obj['bounds'])
            if iou > best_iou:
                best_iou = iou
                best_match = past_id

        # IoU threshold to consider a match
        if best_iou > 0.3:  # Example threshold, can be adjusted
            detection_id = best_match
        else:
            detection_id = next_id
            next_id += 1

        detection_with_id = detection.copy()
        detection_with_id['id'] = detection_id
        current_frame_objects[detection_id] = detection_with_id

    # Append the current frame objects with IDs to history
    history.append(current_frame_objects)

    # Return the current frame objects with their assigned IDs in the required list format
    return list(current_frame_objects.values()), history, next_id


if __name__ == "__main__":
    """
    Usage: demo_multiprocess.py stream_url 
    """

    n_producers = 2
    pixel_mask = 0 # 0, 1 or 2

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
        "../apps/models/yolo_sample_data_day1.pt",
    ]
    yolo_confidence_filters = [
        dict(pedestrian=0.7, car=0.80, van=0.4, truck=0.6, military_tank=0.75, military_truck=0.6, military_vehicle=0.6),
        #dict(military_tank=0.05, military_truck=0.05),
        {"BMP-1": 0.4,
        "Rosomak": 0.4,
        "T72": 0.4,
        "car": 0.4,
        "military_vehicle": 0.4,
        "people": 0.4,
        "soldier": 0.4,
        "trench": 0.4
        }
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
        args=(queue_aggr2cons, pixel_mask, False, yolo_confidence_filters))
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
