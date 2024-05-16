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
    model.set_classes(["car", "pedestrian crossing", "train", "ship"])
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    frmt = "png"
    name = f"yolow_img_{timestamp}.{frmt}"
    path = f"images/{name}"


    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64_image))

    im = Image.open(path)
    img_array = np.asarray(im)

    # Checking number of channels
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]  # Drop the last channel (alpha channel)

    writeLog("logs_yolow.txt", "yolow - Image shape: " + str(np.shape(img_array)))
    # Detection
    result = results = model.predict(img_array)

    end_time = time.time()
    elapsed_time = end_time - start_time
    writeLog("logs_yolow.txt", "yolow - Inference time: " + str(elapsed_time) + " seconds")

    detections = sv.Detections.from_ultralytics(results[0])
    annotated_frame = bounding_box_annotator.annotate(
        scene = img_array.copy(),
        detections = detections[detections.confidence > 0.001]
    )
    #annotated_frame = label_annotator.annotate(
    #    scene=annotated_frame, detections = detections
    #)
    polygon_annotator = sv.PolygonAnnotator()
    annotated_frame = polygon_annotator.annotate(
    scene=annotated_frame,
    detections=detections
    )

    #objects = getObjects("yolow", model, result)

    resDicts = {}
    #resDicts["objects"] = objects
    #resDicts["all_classes"] = model.CLASSES

    return resDicts
