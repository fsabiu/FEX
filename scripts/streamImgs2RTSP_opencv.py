from datetime import datetime
from time import sleep, time
import os
import cv2
import numpy as np

# Custom sorting function
def sort_by_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def create_writer(rtsp_url,width,height,fps):
     # Define VideoWriter properties
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # Codec for the video stream
    
    # Create VideoWriter object
    out = cv2.VideoWriter('appsrc ! videoconvert' + \
        # ' !videorate max-rate=4 ' + \
        ' ! video/x-raw,format=I420' + \
        ' ! x264enc speed-preset=medium bitrate=800 key-int-max=' + str(fps*2) + \
        ' ! video/x-h264,profile=baseline' + \
        ' ! rtspclientsink location=' + rtsp_url,
        cv2.CAP_GSTREAMER, fourcc, fps, (width, height), True)
    if not out.isOpened():
        raise Exception("Can't open video writer")
    return out

def stream_images_as_rtsp(image_folder, rtsp_url, width, height, fps):

    out = create_writer(rtsp_url,width,height,fps)

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files using the custom sorting function
    image_files.sort(key=sort_by_number)
    
    # Write frames to video stream
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            frame_resized = cv2.resize(frame, (width, height))
            out.write(frame_resized)

    # Release VideoWriter object
    out.release()
    print("RTSP streaming finished.")

# Example usage
image_folder = "/home/ubuntu/shared/data/frame_20240517_101530"
rtsp_url = "rtsp://localhost:8554/mystream"
width = 1280
height = 720
fps = 24

stream_images_as_rtsp(image_folder, rtsp_url, width, height, fps)