import cv2
import os
from datetime import datetime
import sys 
import time

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

def save_video_from_stream(stream_url, output_folder, delay):


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    while True:
        
        cap = open_rtsp_stream(stream_url)
        if cap is None:
            print(f"Streaming not active! Retrying in 1 seconds...")
            time.sleep(1)
            continue

        # Set video resolution
        cap.set(3, 1280)
        cap.set(4, 720)

        # Set video codec and get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_folder}/stream_{timestamp}.mp4"

        start_time = datetime.now()
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        while (datetime.now() - start_time).total_seconds() < delay:
            
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                out.write(frame)
        out.release()
        print(f"Video saved to {output_filename}")

        release_rtsp_stream(cap)
        

# Example usage:
def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <stream_url> <output_folder> <delay>")
        return
    # stream_url = 'rtsp://rtspstream:e13d6ca8e1ce8e3c913d7555c48342e4@zephyr.rtsp.stream/movie'
    # video_output_location = 'stream'
    # delay = 10
    stream_url = sys.argv[1]
    video_output_location = sys.argv[2]
    delay_str = sys.argv[3]

    if not delay_str.isdigit():
        print("Error: Delay parameter must be an integer.")
        return

    delay = int(delay_str)
    save_video_from_stream(stream_url, video_output_location,delay)

if __name__ == "__main__":
    main()
