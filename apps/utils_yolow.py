from ultralytics import YOLO
import base64
import cv2
import datetime
from datetime import datetime
import supervision as sv
import os
import numpy as np
from PIL import Image
import time
import torch
#from google.colab.patches import cv2_imshow


def load_yolow():
    # Create the 'models' folder if it doesn't exist
    if not os.path.exists('models'):
        os.mkdir('models')
    
    model = YOLO('models/yolov8l-world.pt') 

    return model

def writeLog(path, obj):
    date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    with open(path, "a") as logfile:
        logfile.write(date_time + ":" + str(obj) + "\n")

def detect_yolow(model, b64_image, confidence):
    classes = ["car", "train"]

    model.set_classes(classes)
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    frmt = "png"
    name = f"yolow_img_{timestamp}.{frmt}"
    path = f"images/{name}"


    # if b64_image is an image write it
    if not b64_image.startswith("/shared"):
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64_image))
    else: # if b64_image is a path read it
        path = b64_image

    im = Image.open(path)
    img_array = np.asarray(im)

    # Checking number of channels
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]  # Drop the last channel (alpha channel)

    # Detection
    results = model.predict(img_array)

    end_time = time.time()
    elapsed_time = end_time - start_time

    detections = sv.Detections.from_ultralytics(results[0])
    annotated_frame = bounding_box_annotator.annotate(
        scene = img_array.copy(),
        detections = detections[detections.confidence > 0.001]
    )

    img = results[0].orig_img
    img_w, img_h = get_np_image_size(img)

    boxes = [b.xywh for b in results[0].boxes]
    confidences = [b.conf for b in results[0].boxes]
    class_ids = [int(b[5]) for b in results[0].boxes.data]
    class_names = [classes[class_id] for class_id in class_ids]
    
    if len(class_ids) > 0:
        class_names = [classes[int(class_id)] for class_id in class_ids if class_id is not None]

    objects = []
    for i, box in enumerate(boxes):
        obj = xywh2xiyi(box, img_w, img_h) # returns dict of bounds: x1 y1, .... x4 y4
        obj["confidence"] = float(confidences[i])
        obj["tagName"] = class_names[i]
        objects.append(obj)

    resDicts = {}
    resDicts["objects"] = objects
    resDicts["all_classes"] = classes
    resDicts["time"] = elapsed_time

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