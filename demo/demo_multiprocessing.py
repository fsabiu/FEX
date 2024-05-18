from collections import defaultdict
from multiprocessing import Process, Queue
import utils_yolo
import time
from ultralytics import YOLO
import cv2

# Function for the producer process
def producer(queue, path, p_id):
    model = YOLO(path)
    cap = cv2.VideoCapture('rtsp://rtspstream:f6182315f67d502cc78abb61e914b860@zephyr.rtsp.stream/movie')

    frames = 0
    total_elapsed = []
    for i in range(200):
        start = time.time()
        frames = frames + 1
        
        ret, frame = cap.read()
        if not ret:
            print("break")
            break

        # Run YOLO prediction on the frame
        results = model.predict(source=frame, device=f'cuda:{p_id}')
        raw_image = results[0].orig_img
        
        res_dict = utils_yolo.get_result_dict(results[0])
        img_hash = hash(frame.tobytes())
        end = time.time()

        queue.put({"yolo_version": p_id, "res_dict": res_dict, "img_hash": img_hash})
        
        elapsed = end - start

        total_elapsed.append(elapsed)

    # Release the capture and close windows
    cap.release()

def consumer(queue_in):
    while True:
        queue_in.get()

# Function for the consumer process
def aggregator(queue_in, queue_out):
    cap = cv2.VideoCapture('rtsp://rtspstream:f6182315f67d502cc78abb61e914b860@zephyr.rtsp.stream/movie')

    detections = defaultdict(lambda: (None, []))

    frames = 0
    total_elapsed = []
    while True:
        start = time.time()
        frames = frames + 1
        
        ret, frame = cap.read()
        if not ret:
            print("break")
            break
        img_hash = hash(frame.tobytes())
        
        detections[img_hash][0] = frames
        
        img_info = queue_in.get()
        
        detections[img_hash].append(img_info)
        if len(detections[img_hash]) == 2:
            print("Ho tutto")
            # plot boxes on image
            queue_out.put(detections[img_hash])
        
        end = time.time()
        elapsed = end - start
        total_elapsed.append(elapsed)
        
        


if __name__ == "__main__":
    queue = Queue()
    output_queue = Queue()
    num_messages = 10

    yolo_1_path = "../apps/models/best_tanks_militaryTrucks.pt"
    producer_process_1 = Process(target=producer, args=(queue, yolo_1_path, 0))
    producer_process_2 = Process(target=producer, args=(queue, yolo_1_path, 1))
    #aggregator_process = Process(target=aggregator, args=(queue, output_queue))
    #consumer_process = Process(target=consumer, args=(output_queue,))


    producer_process_1.start()
    producer_process_2.start()
    #consumer_process.start()

    #consumer_process.join()
    producer_process_1.join()
