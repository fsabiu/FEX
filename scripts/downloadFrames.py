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

def save_frames_from_stream(stream_url, output_folder):
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
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_subfolder}/frame_{frame_count}.jpg", frame)
        frame_count += 1

    release_rtsp_stream(cap)
    print(f"{frame_count} frames saved to {output_folder}")

# Example usage:
def main():
    stream_url = 'rtsp://rtspstream:e13d6ca8e1ce8e3c913d7555c48342e4@zephyr.rtsp.stream/movie'
    frames_output_folder = 'frames'

    # save_video_from_stream(stream_url, video_output_location)
    save_frames_from_stream(stream_url, frames_output_folder)

if __name__ == "__main__":
    main()
