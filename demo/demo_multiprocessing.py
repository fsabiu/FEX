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
def aggregator(queue_in, queue_out, num_producers):
    cap = cv2.VideoCapture('rtsp://rtspstream:f6182315f67d502cc78abb61e914b860@zephyr.rtsp.stream/movie')

    detections = defaultdict(lambda: {"sequence_number": None, "frame": None, "res_dicts": []})
    frame_order = deque()
    lock = Lock()
    
    frames = 0
    total_elapsed = []
    max_queue_size = 200  # Maximum number of frames to keep in memory to prevent memory issues

    def read_frames():
        nonlocal frames
        while True:
            ret, frame = cap.read()
            if not ret:
                print("break")
                break

            img_hash = hash(frame.tobytes())
            with lock:
                if len(frame_order) >= max_queue_size:
                    old_hash = frame_order.popleft()
                    if old_hash in detections:
                        del detections[old_hash]
                frame_order.append(img_hash)
                detections[img_hash]["frame"] = frame
                detections[img_hash]["sequence_number"] = frames
                frames += 1
                if len(detections[img_hash]['res_dicts']) == num_producers:
                    process_frame(img_hash)

    def read_messages():
        while True:
            img_info = queue_in.get()
            img_hash = img_info["img_hash"]
            with lock:
                if img_hash in detections:
                    detections[img_hash]["res_dicts"].append(img_info["res_dict"])
                    if len(detections[img_hash]["res_dicts"]) == num_producers:
                        process_frame(img_hash)

    def process_frame(img_hash):
        img_info = detections[img_hash]
        if img_info["frame"] is not None and len(img_info["res_dicts"]) == num_producers:
            print(f"Processing frame: {img_info['sequence_number']}, Hash: {img_hash}, Results: {img_info['res_dicts']}")
            queue_out.put({
                "sequence_number": img_info["sequence_number"],
                "img_hash": img_hash,
                "res_dicts": img_info["res_dicts"]
            })
            del detections[img_hash]
        else:
            print("Not received 2")

    frame_thread = Thread(target=read_frames)
    message_thread = Thread(target=read_messages)

    frame_thread.start()
    message_thread.start()

    frame_thread.join()
    message_thread.join()

    cap.release()
        
        


if __name__ == "__main__":
    queue = Queue()
    output_queue = Queue()
    num_messages = 10
    n_producers = 2
    yolo_1_path = "../apps/models/best_tanks_militaryTrucks.pt"
    producer_process_1 = Process(target=producer, args=(queue, yolo_1_path, 0))
    producer_process_2 = Process(target=producer, args=(queue, yolo_1_path, 1))
    aggregator_process = Process(target=aggregator, args=(queue, output_queue, n_producers))
    #consumer_process = Process(target=consumer, args=(output_queue,))


    producer_process_1.start()
    producer_process_2.start()
    #consumer_process.start()

    #consumer_process.join()
    producer_process_1.join()
