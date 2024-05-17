import cv2
import os
from datetime import datetime
import sys
import time

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

def save_frames_from_stream(stream_url, output_folder, n):
    
    while True:
        cap = open_rtsp_stream(stream_url)
        if cap is None:
            continue

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subfolder = f"{output_folder}/frame_{timestamp}"
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        num_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream closed. Reopening...")
                break
            
            num_frames += 1

            # check if it's time to save the image
            if num_frames % n == 0:
                # write image to file
                cv2.imwrite(os.path.join(output_subfolder, f'frame_{num_frames}.jpg'), frame)
                print("Saved frame:", num_frames)

        release_rtsp_stream(cap)
        print(f"{num_frames} frames saved to {output_folder}")

# Example usage:
def main():

    if len(sys.argv) < 4:
        print("Usage: python script.py <stream_url> <output_folder> <n>")
        return
    
    stream_url = sys.argv[1]
    frames_output_folder = sys.argv[2]
    n_str = sys.argv[3]

    if not n_str.isdigit():
        print("Error: n parameter must be an integer.")
        return

    n = int(n_str)
    
    save_frames_from_stream(stream_url, frames_output_folder, n)

if __name__ == "__main__":
    main()
