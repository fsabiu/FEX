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

def detect_orcnn(img_path, filter_dict):
    # filter = {
    #         "classes": ["car", "train"]
    #         "confidence": 0.001
    #     },
    print(f"Calling detect orcnn")
    url = 'https://10.0.0.204:2053/detect_orcnn' # fex-6
    payload = json.dumps({"imageData": img_path, "filter": filter_dict})
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

def detect_yolow(img_path, filter_dict):
    # filter = {
    #         "classes": ["car", "train"]
    #         "confidence": 0.001
    #     },
    print(f"Calling detect yolow")
    url = 'https://10.0.0.204:2054/detect_yolow' # fex-6
    payload = json.dumps({"imageData": img_path, "filter": filter_dict})
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


def detect_yolo_tanks_trucks(img_path, filter_dict):
    # filter = {
    #         "classes": ["car", "train"]
    #         "confidence": 0.001
    #     },
    print(f"Calling detect yolow")
    url = 'https://10.0.0.68:2055/detect_yolo_tanks_trucks' # fex-3
    payload = json.dumps({"imageData": img_path, "filter": filter_dict})
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

def update_annotations(img_path, num_frames, result_filter):
    # Collect the results

    # Run detection functions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_orcnn = executor.submit(detect_orcnn, img_path, result_filter["orcnn"])
        future_yolow = executor.submit(detect_yolow, img_path, result_filter["yolow"])
        future_yolo_tanks_trucks = executor.submit(detect_yolo_tanks_trucks, img_path, result_filter["yolo_tanks_trucks"])

        
        orcnn_results = future_orcnn.result()
        yolow_results = future_yolow.result()
        yolo_tanks_trucks = future_yolow.result()

    # Aggregate results (example: concatenate the results)
    annotations = {
        "orcnn": orcnn_results,
        "yolow": yolow_results,
        "yolo_tanks_trucks": yolo_tanks_trucks
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
    
    result_filter = {
        "yolow": {
            "classes": ["car", "train"],
            "confidence": 0.001
        },
        "orcnn":{
            "classes": ["plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "storage-tank", "roundabout", "harbor", "helicopter"],
            "confidence": 0.6
        },
        "yolo_tanks_trucks":{
            "classes": ["military_tank", "military_truck"],
            "confidence": 0.6
        }
    }

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
                for model in last_annotations.keys():
                    if "time" in last_annotations[model]:
                        print("Time model " + model + " : " + str(last_annotations[model]["time"]))
                    else:
                        print(last_annotations[model])
                # update last annotation
                previous_annotations = last_annotations
                
                # Save the image
                cv2.imwrite(img_path, frame)

                # Call detection asynchronously
                print(f"Frame {num_frames}: running annotation thread")
                detection_thread = threading.Thread(target=update_annotations, args=(img_path, num_frames, result_filter))
                detection_thread.start()
            else:
                print("Sending frame " + str(num_frames))

            
            # Initialize an empty list to store all objects
            ensemble_annotations = []

            # Iterate over each dictionary in the list
            for model in last_annotations.keys():
                # Extract the "objects" list from the current dictionary
                objects_list = last_annotations[model]["objects"]
                # Extend the merged_objects list with the objects_list
                ensemble_annotations.extend(objects_list)

            pil_image = Image.fromarray(frame)
            pil_image = annotate_img(pil_image, ensemble_annotations)
            pil_image.save(img_path_det)

    release_rtsp_stream(cap)

from PIL import Image, ImageDraw

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


# Example usage:
def main():
    """
    Usage: demo_maindetect.py stream_url output_folder
    """
    # stream_url = 'rtsp://rtspstream:e13d6ca8e1ce8e3c913d7555c48342e4@zephyr.rtsp.stream/movie'
    # frames_output_folder = 'frames'

    stream_url = sys.argv[1]
    output_folder = sys.argv[2]

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    process_stream(stream_url, output_folder)

if __name__ == "__main__":
    main()