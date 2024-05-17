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

    writeLog("logs_yolow.txt", "yolow - Image shape: " + str(np.shape(img_array)))
    # Detection
    results = model.predict(img_array)

    end_time = time.time()
    elapsed_time = end_time - start_time
    writeLog("logs_yolow.txt", "yolow - Inference time: " + str(elapsed_time) + " seconds")
    #writeLog("logs_yolow.txt", type(results[0]))
    
    # if result[0]["obb"] is not None:

    #     writeLog("logs_yolow.txt", "results[0]")
    #     writeLog("logs_yolow.txt", results[0])

    detections = sv.Detections.from_ultralytics(results[0])
    annotated_frame = bounding_box_annotator.annotate(
        scene = img_array.copy(),
        detections = detections[detections.confidence > 0.001]
    )

    img = results[0].orig_img
    img_w, img_h = get_np_image_size(img)

    boxes = [b.xywh for b in results[0].boxes]
    confidences = [b.conf for b in results[0].boxes]
    class_ids = [b.id for b in results[0].boxes if b.id]
    class_names = []
    
    if len(class_ids) > 0:
        class_names = [classes[int(class_id)] for class_id in class_ids if class_id is not None]


    #if class_ids != []:
    #writeLog("logs_yolow.txt", results[0])
    if len(boxes)>0:
        writeLog("logs_yolow.txt", boxes)
        writeLog("logs_yolow.txt", confidences)
        writeLog("logs_yolow.txt", class_ids)
        writeLog("logs_yolow.txt", class_names)

    objects = []
    for i, box in enumerate(boxes):
        obj = xywh2xiyi(box, img_w, img_h) # returns dict of bounds: x1 y1, .... x4 y4
        obj["confidence"] = confidences[i]
        obj["tagName"] = "TO DO"
        objects.append(obj)
    
    # 2024/05/17 10:07:55:Detections(xyxy=array([[     618.06,      494.75,      858.95,      555.18],
    #    [     284.33,      618.58,         319,       649.8],
    #    [     1051.8,      584.13,      1093.1,       614.6],
    #    [     969.94,      673.32,      1011.7,      706.84],
    #    [     324.88,      676.43,      354.35,      718.55],
    #    [     363.67,      653.16,      390.53,      696.26]], dtype=float32), mask=None, confidence=array([    0.59864,     0.43561,     0.38763,     0.36425,       0.283,     0.25649], dtype=float32), class_id=array([1, 0, 0, 0, 0, 0]), tracker_id=None, data={'class_name': array(['train', 'car', 'car', 'car', 'car', 'car'], dtype='<U5')})
    
    #annotated_frame = label_annotator.annotate(
    #    scene=annotated_frame, detections = detections
    #)
    # polygon_annotator = sv.PolygonAnnotator()
    # annotated_frame = polygon_annotator.annotate(
    # scene=annotated_frame,
    # detections=detections
    # )

    resDicts = {}
    resDicts["objects"] = objects
    resDicts["all_classes"] = classes
    writeLog(resDicts, class_names)
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