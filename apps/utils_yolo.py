from ultralytics import YOLO
import base64
import cv2
import datetime
from datetime import datetime
import supervision as sv
import os
import numpy as np
from PIL import Image
import torch
import time
#from google.colab.patches import cv2_imshow


def load_yolo(model):
    # Create the 'models' folder if it doesn't exist
    if not os.path.exists('models'):
        os.mkdir('models')
    
    model = YOLO('models/'+str(model)) #set this up with the trained yolo model

    return model

def writeLog(path, obj):
    date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    with open(path, "a") as logfile:
        logfile.write(date_time + ":" + str(obj) + "\n")

def detect_yolo(model, b64_image, confidence):
    #model.set_classes(["car", "pedestrian crossing", "train", "ship"])
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    #start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    frmt = "png"
    name = f"yolo_img_{timestamp}.{frmt}"
    path = f"images/{name}"


    # if b64_image is an image write it
    if not b64_image.startswith("/shared"):
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64_image))
    else: # if b64_image is a path read it
        path = b64_image

    #im = Image.open(path)
    #img_array = np.asarray(im)

    # Checking number of channels
    #if img_array.shape[2] == 4:
    #    img_array = img_array[:, :, :3]  # Drop the last channel (alpha channel)

    #writeLog("logs_yolo.txt", "yolo - Image shape: " + str(np.shape(img_array)))
    # Detection
    start_time = time.time()
    results = model(path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    writeLog("logs_yolo.txt", "yolo - Inference time: " + str(elapsed_time) + " seconds")

    #detections = sv.Detections.from_ultralytics(results[0])
    #annotated_frame = bounding_box_annotator.annotate(
    #    scene = img_array.copy(),
    #    detections = detections[detections.confidence > 0.001]
    #)
    #annotated_frame = label_annotator.annotate(
    #    scene=annotated_frame, detections = detections
    #)
    #polygon_annotator = sv.PolygonAnnotator()
    #annotated_frame = polygon_annotator.annotate(
    #scene=annotated_frame,
    #detections=detections
    #)

    #objects = getObjects("yolow", model, result)

    # returns objects, all classes, class_ids. objects is a list of bounds, tagName, confidence
    resDicts = get_result_dict(model, results[0]) 
    resDicts["time"] = elapsed_time
    writeLog("logs_yolo.txt", resDicts)
    return resDicts

def get_result_dict(model, result):
    img = result.orig_img
    img_w, img_h = get_np_image_size(img)

    boxes = [b.xywh for b in result.boxes]
    confidences = [b.conf for b in result.boxes]
    class_ids = [int(b[5]) for b in result.boxes.data]
    class_names = [model.names[class_id] for class_id in class_ids]

    objects = []
    for i, box in enumerate(boxes):
        obj = xywh2xiyi(box, img_w, img_h) # returns dict of bounds: x1 y1, .... x4 y4
        obj["confidence"] = float(confidences[i])
        obj["tagName"] = class_names[i]
        objects.append(obj)

    resDicts = {}
    resDicts["objects"] = objects
    resDicts["all_classes"] = class_names

    return resDicts

def get_np_image_size(image):
    if image.ndim == 3:  # Color image
        height, width, channels = image.shape
    elif image.ndim == 2:  # Grayscale image
        height, width = image.shape
        channels = 1  # Grayscale image has 1 channel
    else:
        raise ValueError("Unsupported image dimensions")
    
    return width, height

def xywh2xiyi(xywh, img_w, img_h):
    x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = 0

    if isinstance(xywh, torch.Tensor):
        x_center = float(xywh[0, 0])
        y_center = float(xywh[0, 1])
        width = float(xywh[0, 2])
        height = float(xywh[0, 3])

        # Convert relative coordinates to absolute coordinates
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center - height / 2) * img_h)
    x3 = int((x_center + width / 2) * img_w)
    y3 = int((y_center + height / 2) * img_h)
    x4 = int((x_center - width / 2) * img_w)
    y4 = int((y_center + height / 2) * img_h)

    bounds = {
        "bounds": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x3": x3,
            "y3": y3,
            "x4": x4,
            "y4": y4
        }
    }

    return bounds
