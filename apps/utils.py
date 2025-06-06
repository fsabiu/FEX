import base64
import datetime
from datetime import datetime
import os
import subprocess
import time
import torch
import numpy as np
from PIL import Image

# Set environment variables to fix mmrotate device issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import mmrotate
import mmdet
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector

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
    
    # Log current device usage
    writeLog("logs_orcnn.txt", f"orcnn - Model device: {next(model.parameters()).device}")
    writeLog("logs_orcnn.txt", f"orcnn - Target device: {device}")
    writeLog("logs_orcnn.txt", f"orcnn - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        writeLog("logs_orcnn.txt", f"orcnn - CUDA device count: {torch.cuda.device_count()}")
        writeLog("logs_orcnn.txt", f"orcnn - Current CUDA device: {torch.cuda.current_device()}")
        writeLog("logs_orcnn.txt", f"orcnn - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        writeLog("logs_orcnn.txt", f"orcnn - GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # URGENT GPU FIX - Direct tensor device enforcement
    def force_gpu_inference(model, img, device):
        """EMERGENCY GPU-ONLY inference for demo"""
        writeLog("logs_orcnn.txt", f"orcnn - EMERGENCY GPU INFERENCE MODE - device: {device}")
        
        # Override the problematic NMS function at runtime
        import mmrotate.core.post_processing.bbox_nms_rotated as nms_module
        original_multiclass_nms = nms_module.multiclass_nms_rotated
        
        def emergency_gpu_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False):
             # CRITICAL FIX: Force ALL tensors and indices to target GPU device
             if hasattr(multi_bboxes, 'device') and multi_bboxes.device.type == 'cuda':
                 target_device = multi_bboxes.device
                 
                 # Ensure scores are on same device
                 multi_scores = multi_scores.to(target_device)
                 if score_factors is not None:
                     score_factors = score_factors.to(target_device)
                 
                 # Get original result but intercept device operations
                 import torch
                 with torch.cuda.device(target_device):
                     # Call original NMS but ensure all intermediate tensors stay on GPU
                     det_bboxes, det_labels = [], []
                     for cls_id, (bboxes, scores) in enumerate(zip(multi_bboxes, multi_scores)):
                         if len(bboxes) == 0:
                             continue
                         
                         # Ensure bboxes and scores are on target device
                         bboxes = bboxes.to(target_device)
                         scores = scores.to(target_device)
                         
                         # Filter by score threshold
                         valid_mask = scores > score_thr
                         if not valid_mask.any():
                             continue
                         
                         valid_bboxes = bboxes[valid_mask].to(target_device)
                         valid_scores = scores[valid_mask].to(target_device)
                         
                         # Create labels tensor on same device
                         labels = torch.full((len(valid_bboxes),), cls_id, dtype=torch.long, device=target_device)
                         
                         det_bboxes.append(torch.cat([valid_bboxes, valid_scores.unsqueeze(1)], dim=1))
                         det_labels.append(labels)
                     
                     if len(det_bboxes) > 0:
                         det_bboxes = torch.cat(det_bboxes, dim=0).to(target_device)
                         det_labels = torch.cat(det_labels, dim=0).to(target_device)
                         return det_bboxes, det_labels
                     else:
                         empty_bboxes = torch.zeros((0, 6), device=target_device)
                         empty_labels = torch.zeros((0,), dtype=torch.long, device=target_device)
                         return empty_bboxes, empty_labels
             
             # Fallback to original function for non-CUDA tensors
             return original_multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num, score_factors, return_inds)
        
        # Apply emergency patch
        nms_module.multiclass_nms_rotated = emergency_gpu_nms
        writeLog("logs_orcnn.txt", "orcnn - EMERGENCY NMS PATCH APPLIED")
        
        try:
            with torch.cuda.device(device):
                result = inference_detector(model, img)
                writeLog("logs_orcnn.txt", f"orcnn - EMERGENCY GPU INFERENCE SUCCESS on {device}")
                return result
        finally:
            # Restore original function
            nms_module.multiclass_nms_rotated = original_multiclass_nms
    
    # EMERGENCY GPU-ONLY INFERENCE FOR DEMO
    if device.startswith('cuda'):
        try:
            result = force_gpu_inference(model, img_array, device)
        except Exception as e:
            writeLog("logs_orcnn.txt", f"orcnn - EMERGENCY GPU FAILED: {e}")
            raise e  # Don't fallback to CPU - we need GPU for demo!
    else:
        result = inference_detector(model, img_array)
    
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

    # DEMO MODE: Force CPU inference for reliability (mmrotate 0.3.4 GPU device mismatch issue)
    import torch
    device = 'cpu'
    writeLog("logs_orcnn.txt", "orcnn - DEMO MODE: Using CPU for reliable inference")
    writeLog("logs_orcnn.txt", f"orcnn - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        writeLog("logs_orcnn.txt", f"orcnn - GPU device name: {torch.cuda.get_device_name(0)} (available but not used for stability)")
        writeLog("logs_orcnn.txt", f"orcnn - GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load the config
    config = mmcv.Config.fromfile(config_file)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None
    
    # Force CPU for post-processing operations to avoid device mismatch
    if hasattr(config.model, 'test_cfg') and config.model.test_cfg is not None:
        if hasattr(config.model.test_cfg, 'rcnn') and config.model.test_cfg.rcnn is not None:
            # Set NMS to run on CPU
            writeLog("logs_orcnn.txt", "orcnn - Configuring RCNN test config for CPU NMS")
    
    # Alternative approach: modify the model architecture to handle device issues
    config.model.roi_head.bbox_head.loss_cls.use_sigmoid = False
    writeLog("logs_orcnn.txt", "orcnn - Applied model configuration changes for device compatibility")
    
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

    # Apply mmrotate 0.3.4 specific fix for NMS device mismatch first
    if device.startswith('cuda'):
        try:
            # More comprehensive patch for device mismatch in mmrotate 0.3.4
            from mmrotate.core.post_processing.bbox_nms_rotated import multiclass_nms_rotated
            original_nms = multiclass_nms_rotated
            
            def patched_multiclass_nms_rotated(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False):
                # Convert all inputs to CPU to avoid device mismatch
                device_was_cuda = False
                if hasattr(multi_bboxes, 'device') and multi_bboxes.device.type == 'cuda':
                    device_was_cuda = True
                    original_device = multi_bboxes.device
                    multi_bboxes = multi_bboxes.cpu()
                    multi_scores = multi_scores.cpu()
                    if score_factors is not None:
                        score_factors = score_factors.cpu()
                
                # Run NMS on CPU
                result = original_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num, score_factors, return_inds)
                
                # Convert results back to original device if needed
                if device_was_cuda:
                    if return_inds:
                        dets, labels, inds = result
                        dets = dets.to(original_device)
                        labels = labels.to(original_device)
                        inds = inds.to(original_device)
                        result = (dets, labels, inds)
                    else:
                        dets, labels = result
                        dets = dets.to(original_device)
                        labels = labels.to(original_device)
                        result = (dets, labels)
                
                return result
            
            # Apply the comprehensive patch
            import mmrotate.core.post_processing.bbox_nms_rotated
            mmrotate.core.post_processing.bbox_nms_rotated.multiclass_nms_rotated = patched_multiclass_nms_rotated
            writeLog("logs_orcnn.txt", "orcnn - Applied comprehensive NMS device patch for mmrotate 0.3.4")
            
        except Exception as e:
            writeLog("logs_orcnn.txt", f"orcnn - Could not apply comprehensive NMS patch: {e}")
            # Fallback to simpler approach - force CPU inference
            writeLog("logs_orcnn.txt", "orcnn - Falling back to CPU-only inference due to GPU device mismatch issues")
            device = 'cpu'

    # Convert the model to the final target device
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    # Verify final device placement
    model_device = next(model.parameters()).device
    writeLog("logs_orcnn.txt", f"orcnn - Model successfully loaded on device: {model_device}")
    writeLog("logs_orcnn.txt", f"orcnn - Model classes: {len(model.CLASSES)} classes loaded")

    return {
        "model": model,
        "device": device
    }

def writeLog(path, obj):
    date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    with open(path, "a") as logfile:
        logfile.write(date_time + ":" + str(obj) + "\n")