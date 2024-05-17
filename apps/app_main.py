import concurrent.futures
import cv2
import os
from datetime import datetime
import json
from PIL import Image
import requests
import threading
import urllib3
import sys
import time

last_annotations = {}

def open_rtsp_stream(stream_url):
    cap = None
    while cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Waiting for stream ...")
            time.sleep(1)
    return cap

def release_rtsp_stream(cap):
    if cap:
        cap.release()
        cv2.destroyAllWindows()

def detect_orcnn(img_path, filter):
    # filter = {
    #         "classes": ["car", "train"]
    #         "confidence": 0.001
    #     },
    print(f"Calling detect orcnn")
    url = 'https://10.0.0.204:2053/detect_orcnn' # fex-6
    payload = json.dumps({"imageData": img_path, "filter": filter})
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    ret = {}
    # Parse the response content to JSON
    try:
        ret = response.json()
    except ValueError as e:
        print(f"Failed to parse JSON from {url}: {e}")
        print(f"Response content: {response.content}")

    return ret

def detect_yolow(img_path, filter):
    # filter = {
    #         "classes": ["car", "train"]
    #         "confidence": 0.001
    #     },
    print(f"Calling detect yolow")
    url = 'https://10.0.0.204:2054/detect_yolow' # fex-6
    payload = json.dumps({"imageData": img_path, "filter": filter})
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    ret = {}
    # Parse the response content to JSON
    try:
        ret = response.json()
    except ValueError as e:
        print(f"Failed to parse JSON from {url}: {e}")
        print(f"Response content: {response.content}")

    return ret

def annotate_img(pil_image):
    return pil_image

def update_annotations(img_path, num_frames):
    # Collect the results
    result_filter = {
        "yolow": {
            "classes": ["car", "train"],
            "confidence": 0.001
        },
        "orcnn":{
            "classes": ["plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "storage-tank", "roundabout", "harbor", "helicopter"],
            "confidence": 0.6
        }
    }

    # Run detection functions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_orcnn = executor.submit(detect_orcnn, img_path, result_filter["orcnn"])
        future_yolow = executor.submit(detect_yolow, img_path, result_filter["yolow"])

        
        orcnn_results = future_orcnn.result()
        yolow_results = future_yolow.result()

    # Aggregate results (example: concatenate the results)
    annotations = {
        "orcnn": orcnn_results,
        "yolow": yolow_results
    }

    global last_annotations
    last_annotations = annotations
    print(f"Annotations uptated with frame {num_frames}")

    return


def process_stream(stream_url, output_folder):
    global last_annotations
    previous_annotations = None
    
    cap = open_rtsp_stream(stream_url)

    num_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        num_frames += 1
        
        if frame is not None:

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            img_path = os.path.join(output_folder, 'frame_{}.jpg'.format(timestamp))
            img_path_det = os.path.join(output_folder, 'det_frame_{}.jpg'.format(timestamp))

            
            # Saving image each n frames
            if previous_annotations != last_annotations:
                
                # update last annotation
                previous_annotations = last_annotations
                
                # Save the image
                cv2.imwrite(img_path, frame)

                # Call detection asynchronously
                print(f"Frame {num_frames}: running annotation thread")
                detection_thread = threading.Thread(target=update_annotations, args=(img_path, num_frames))
                detection_thread.start()
            else:
                print("Sending frame " + str(num_frames))

            
            pil_image = Image.fromarray(frame)
            pil_image = annotate_img(pil_image)
            pil_image.save(img_path_det)

    release_rtsp_stream(cap)

# Example usage:
def main():
    """
    Usage: demo_maindetect.py stream_url output_folder
    """
    # stream_url = 'rtsp://rtspstream:e13d6ca8e1ce8e3c913d7555c48342e4@zephyr.rtsp.stream/movie'
    # frames_output_folder = 'frames'

    stream_url = sys.argv[1]
    frames_output_folder = sys.argv[2]

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # stream processing
    process_stream(stream_url, frames_output_folder)

if __name__ == "__main__":
    main()