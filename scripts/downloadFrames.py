import cv2
import os
from datetime import datetime
import sys

def open_rtsp_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open the video stream.")
        return None
    return cap

def release_rtsp_stream(cap):
    if cap:
        cap.release()
        cv2.destroyAllWindows()

def save_frames_from_stream(stream_url, output_folder,n):
    cap = open_rtsp_stream(stream_url)
    if cap is None:
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subfolder = f"{output_folder}/frame_{timestamp}"
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    
    # initialize variables for FPS calculation
    num_frames = 0
    interval = 1  # interval for FPS calculation in seconds
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        num_frames += 1

        # check if it's time to save the image
        if num_frames % n == 0:
            # write image to file
            cv2.imwrite(os.path.join(output_subfolder, 'frame_{}.jpg'.format(num_frames)), frame)
            print("Saved frame:", num_frames)

    release_rtsp_stream(cap)
    print(f"{frame_count} frames saved to {output_folder}")

# Example usage:
def main():
   
    # stream_url = 'rtsp://rtspstream:e13d6ca8e1ce8e3c913d7555c48342e4@zephyr.rtsp.stream/movie'
    # frames_output_folder = 'frames'

    stream_url = sys.argv[1]
    frames_output_folder = sys.argv[2]
    n_str = sys.argv[3]

    if not n_str.isdigit():
        print("Error: Delay parameter must be an integer.")
        return

    n = int(n_str)
    
    save_frames_from_stream(stream_url, frames_output_folder, n)

if __name__ == "__main__":
    main()
