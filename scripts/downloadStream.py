import cv2
import os
from datetime import datetime

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

def save_video_from_stream(stream_url, output_location):
    cap = open_rtsp_stream(stream_url)
    if cap is None:
        return

    # Create folder if not exists
    output_folder = output_location
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Set video resolution
    cap.set(3, 1280)
    cap.set(4, 720)

    # Set video codec and get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_folder}/stream_{timestamp}.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    release_rtsp_stream(cap)
    print(f"Video saved to {output_location}")

# Example usage:
def main():
    stream_url = 'rtsp://rtspstream:e13d6ca8e1ce8e3c913d7555c48342e4@zephyr.rtsp.stream/movie'
    video_output_location = 'stream'

    save_video_from_stream(stream_url, video_output_location)

if __name__ == "__main__":
    main()
