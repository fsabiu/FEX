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

# Function for the producer process
def producer(stream_url, queue, path, p_id):
    model = YOLO(path)
    cap = cv2.VideoCapture(stream_url)

    frames = 0
    total_elapsed = []
    while True:
        start = time.time()
        frames = frames + 1
        
        ret, frame = cap.read()
        if not ret:
            #print("break")
            break

        # Run YOLO prediction on the frame
        results = model.predict(source=frame, device=f'cuda:{p_id}', verbose=False)
        raw_image = results[0].orig_img
        
        res_dict = utils_yolo.get_result_dict(model, results[0])
        img_hash = hash(frame.tobytes())
        end = time.time()

        #print(f'\nProducer {p_id} - Sending detections')
        res_dict["yolo_producer"] = p_id
        
        queue.put({"yolo_producer": p_id, "res_dict": res_dict, "img_hash": img_hash})
        
        elapsed = end - start

        total_elapsed.append(elapsed)

    # Release the capture and close windows
    cap.release()

def consumer(queue_in, error_queue, confidence_filters):
    # In queue we have
    # "frame": img_info["frame"],
    # "sequence_number": img_info["sequence_number"],
    # "img_hash": img_hash,
    # "res_dicts": img_info["res_dicts"]
    
    while True:
        try:
            processed_frame = queue_in.get()
            #print("\nConsumer")
            ##print(f'Consumed frame {processed_frame["sequence_number"]}')
            ##print(processed_frame["res_dicts"])
            ##print(confidence_filters)
            #filtered_objects = get_filtered_objects(processed_frame["res_dicts"], confidence_filters)
                        
            #annotations = [d.objects for d in filtered_objects]
            #flattened_annotations = list(itertools.chain.from_iterable(annotations))
            
            #pil_image = Image.fromarray(processed_frame["frame"])
            #img_res = annotate_img(processed_frame["frame"], annotations)
            ##print("Image ready")
        except RuntimeError:
            error_queue.put(traceback.format_exc())

def get_filtered_objects(data, confidence_filters):
    result = []
    for entry in data:
        for obj in entry['objects']: # if confidence filter for that object is below then append
            if 'confidence' in obj and detection["yolo_producer"] and 'tagName' in obj and detection["yolo_producer"] in confidence_filters and obj["tagName"] in confidence_filters:
                if 'confidence' in obj and obj['confidence'] >= confidence_filters[detection["yolo_producer"]][obj["tagName"]]:
                    result.append(obj)
            else:
                pass
                #print("Warning labels confidence")
    return result

def annotate_img(image, annotations):
    """
    Annotate the image with bounding boxes.

    Args:
    - image: PIL image object
    - annotations: List of dictionaries containing annotation bounds

    Returns:
    - Annotated PIL image object
    """
    draw = ImageDraw.Draw(image)
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
        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline="red")

    return image


def annotate_img_opencv(image, annotations):
    """
    Annotate the image with bounding boxes.

    Args:
    - image: PIL image object
    - annotations: List of dictionaries containing annotation bounds

    Returns:
    - Annotated PIL image object
    """
    draw = ImageDraw.Draw(image)
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
        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline="red")

    return image

# Function for the consumer process
def aggregator(stream_url, queue_in, queue_out, num_producers, confidence_filters):
    cap = cv2.VideoCapture(stream_url)

    detections = defaultdict(lambda: {"sequence_number": None, "frame": None, "res_dicts": []})
    lock = RLock()
    
    first_frame_sent = False
    frame_nr = 0
    total_elapsed = []
    max_queue_size = 10  # Maximum number of frames to keep in memory to prevent memory issues
    producer_queues = defaultdict(deque)

    def read_frames():
        nonlocal frame_nr
        while True:
            ret, frame = cap.read()
            frame_nr += 1
            if not ret:
                #print("break")
                break

            img_hash = hash(frame.tobytes())
            with lock:
                #print(f"Read frame - received frame nr {frame_nr}")
                detections[img_hash]["frame"] = frame
                detections[img_hash]["sequence_number"] = frame_nr
                process_frame(img_hash)

    def read_messages():
        while True:
            img_info = queue_in.get()
            #print("Message read")
            img_hash = img_info["img_hash"]
            with lock:
                detections[img_hash]["res_dicts"].append(img_info["res_dict"])
                producer_queues[img_info["yolo_producer"]].append(img_hash)
                #print("Received: " + str(len(detections[img_hash]["res_dicts"])))
                process_frame(img_hash)

    def process_frame(img_hash):
        img_info = detections[img_hash]
        if img_info["frame"] is not None and len(img_info["res_dicts"]) == num_producers:
            
            #print("\nAggregator - Process frame")
            #print(f"Processing frame: {img_info['sequence_number']}, Results: {img_info['res_dicts']}")

            queue_out.put({
                "frame": img_info["frame"],
                "sequence_number": img_info["sequence_number"],
                "img_hash": img_hash,
                "res_dicts": img_info["res_dicts"]
            })
            print(f"Sent frame nr {img_info['sequence_number']}" )

            nonlocal first_frame_sent
            detections_keys = list(detections.keys())
            begin = len(detections_keys)
            for im_hash in detections_keys:
                if detections[im_hash]["sequence_number"] is not None and detections[im_hash]["sequence_number"] <= img_info["sequence_number"]:
                    del detections[im_hash]
            end = len(list(detections.keys()))
            print(f"From {begin} to {end} items in detections")
            first_frame_sent = True

            for id_prods in range(num_producers):
                while len(producer_queues[id_prods]):
                    past_img = producer_queues[id_prods].popleft()
                    if past_img in detections:
                        del detections[past_img]
                    if past_img == img_hash:
                        break
            
    frame_thread = Thread(target=read_frames)
    message_thread = Thread(target=read_messages)

    frame_thread.start()
    message_thread.start()

    frame_thread.join()
    message_thread.join()

    cap.release()
        

if __name__ == "__main__":
    """
    Usage: demo_multiprocess.py stream_url 
    """

    stream_url = sys.argv[1]  

    queue = Queue()
    error_queue = Queue()
    output_queue = Queue()

    n_producers = 2
    yolo_1_path = "../apps/models/best_tanks_militaryTrucks.pt"
    yolo_2_path = "../apps/models/best_tanks_militaryTrucks.pt"

    yolo1_confidence_filters = {
        "military_tank": 0.05,
        "military_truck": 0.05
    }

    yolo2_confidence_filters = {
        "military_tank": 0.05,
        "military_truck": 0.05
    }

    confidence_filters = [yolo1_confidence_filters, yolo2_confidence_filters]

    producer_process_1 = Process(target=producer, args=(stream_url, queue, yolo_1_path, 0))
    producer_process_2 = Process(target=producer, args=(stream_url, queue, yolo_1_path, 1))
    aggregator_process = Process(target=aggregator, args=(stream_url, queue, output_queue, n_producers, confidence_filters))
    consumer_process = Process(target=consumer, args=(output_queue, error_queue, confidence_filters))


    producer_process_1.start()
    producer_process_2.start()
    consumer_process.start()
    aggregator_process.start()

    
    producer_process_1.join()
    producer_process_2.join()
    aggregator_process.join()
    consumer_process.join()

    # Check if there were any exceptions
    if not error_queue.empty():
        #print("An error occurred in the consumer process:")
        with open('logs_ww.txt', 'a') as f:
            f.write("An error occurred in the consumer process:\n")
            f.write(error_queue.get())
            f.write("\n")  # Ensure a new line after the traceback