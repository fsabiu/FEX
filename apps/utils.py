import base64
import datetime
from datetime import datetime
import mmrotate
import mmdet
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector
import numpy as np
import os
import subprocess
from PIL import Image
import time
import torch

def detect_orcnn(model_dict, b64_image, confidence):
    model = model_dict["model"]
    device = model_dict["device"]

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    frmt = "png"
    name = f"orcnn_img_{timestamp}.{frmt}"
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

    writeLog("logs_orcnn.txt", "orcnn - Image shape: " + str(np.shape(img_array)))
    
    # Ensure model is on the correct device and in eval mode
    model.to(device)
    model.eval()
    
    # Detection with error handling for device issues
    try:
        result = inference_detector(model, img_array)
    except RuntimeError as e:
        if "indices should be either on cpu or on the same device" in str(e) or "CUDA" in str(e):
            writeLog("logs_orcnn.txt", f"orcnn - GPU inference failed: {str(e)}, falling back to CPU")
            # Move model to CPU and retry
            model.to('cpu')
            device = 'cpu'
            result = inference_detector(model, img_array)
        else:
            raise e
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    writeLog("logs_orcnn.txt", "orcnn - Inference time: " + str(elapsed_time) + " seconds")

    objects = getObjects("orcnn", model, result)

    resDicts = {}
    resDicts["time"] = elapsed_time
    resDicts["objects"] = objects
    resDicts["all_classes"] = model.CLASSES

    return resDicts

def getObjects(modelname, model, result):
    """
    Input:
    - name of the model (e.g. orcnn)
    - model object
    - detection result
    Returns: list of {bounds, confidence, tagname}
    """
    objects = []

    if modelname == "orcnn":
        bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        bboxes = np.vstack(bbox_result) #boxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or (n, 6).
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        
        for i, bbox in enumerate(bboxes):
            xc, yc, w, h, ag = bbox[:5]
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            #p1 = (xc - wx - hx, yc - wy - hy)
            #p2 = (xc + wx - hx, yc + wy - hy)
            #p3 = (xc + wx + hx, yc + wy + hy)
            #p4 = (xc - wx + hx, yc - wy + hy)
            x1 = xc - wx - hx
            y1 = yc - wy - hy
            x2 = xc + wx - hx
            y2 = yc + wy - hy
            x3 = xc + wx + hx
            y3 = yc + wy + hy
            x4 = xc - wx + hx
            y4 = yc - wy + hy
            
            obj = {
                "bounds": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "x3": float(x3),
                    "y3": float(y3),
                    "x4": float(x4),
                    "y4": float(y4)
                },
                "confidence": float(bbox[5]),
                "tagName": model.CLASSES[labels[i]]
            }

            objects.append(obj)
    return objects

def load_orcnn():
    # Create the 'models' folder if it doesn't exist
    if not os.path.exists('models'):
        os.mkdir('models')
    
    # Create the 'models' folder if it doesn't exist
    if not os.path.exists('images'):
        os.mkdir('images')

    # Check if the files exist, and download them only if they don't
    config_file = './models/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    checkpoint_file = './models/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

    if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
        subprocess.run(['mim', 'download', 'mmrotate', '--config', 'oriented_rcnn_r50_fpn_1x_dota_le90', '--dest', './models'])
    else:
        print("Model and config file already exist.")

    # Set the device to be used for evaluation with fallback
    import torch
    if torch.cuda.is_available():
        device = 'cuda:0'
        writeLog("logs_orcnn.txt", "orcnn - Using GPU for inference")
    else:
        device = 'cpu'
        writeLog("logs_orcnn.txt", "orcnn - CUDA not available, using CPU for inference")

    # Load the config
    config = mmcv.Config.fromfile(config_file)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None
    
    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint with appropriate device mapping
    try:
        checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)
    except Exception as e:
        writeLog("logs_orcnn.txt", f"orcnn - Failed to load on {device}, trying CPU: {str(e)}")
        device = 'cpu'
        checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to the target device
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    return {
        "model": model,
        "device": device
    }

def writeLog(path, obj):
    date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    with open(path, "a") as logfile:
        logfile.write(date_time + ":" + str(obj) + "\n")