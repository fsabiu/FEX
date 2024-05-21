import cv2
import os

def minutes_to_seconds(minutes):
    return minutes * 60

def capture_frames(video_path, intervals, capture_fps):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the frames per second (fps) of the video
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print("Error: Could not retrieve FPS from video file")
        return
    
    # Create output directory based on video path
    base_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_dir = os.path.expanduser(f'~/shared/clean_frames/{name_without_ext}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_index = 1

    # Process each interval
    for start_min, end_min in intervals:
        start_sec = minutes_to_seconds(start_min)
        end_sec = minutes_to_seconds(end_min)
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

        while frame_count <= end_frame:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Warning: Could not read frame at position {frame_count}")
                break

            if int((frame_count - start_frame) % (video_fps // capture_fps)) == 0:
                frame_filename = os.path.join(output_dir, f"frame_{frame_index}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                frame_index += 1

            frame_count += 1

    # Release the video capture object
    video_capture.release()
    print("Frame extraction completed.")

# Example usage
video_path = '/home/ubuntu/shared/Video22.mp4'
intervals = [
    [1.25, 2.07],   # Interval in minutes
    [2.11, 4.50],
    [5.02, 7.27],
    [12.43, 13.16],
    [13.55, 14.25],
]
capture_fps = 2  # Number of frames to capture per second

capture_frames(video_path, intervals, capture_fps)
