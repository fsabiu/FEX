import cv2
import os
from datetime import datetime

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
    today = datetime.now().strftime("%d%m")
    output_dir = os.path.expanduser(f'~/shared/clean_frames/{today}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_index = 1
    frame_counter =0
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
                frame_filename = os.path.join(output_dir, f"{base_name}_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {base_name}_{frame_count}")
                frame_index += 1
    
            frame_count += 1

    # Release the video capture object
    video_capture.release()
    print("Frame extraction completed.")


# Example usage
videos_dict = {
    '/home/ubuntu/shared/stream/stream_20240521_061601.mp4': [
        [1.37, 4.33],
        [4.52, 5.02]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_062303.mp4': [
        [4.59, 5.04]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_062819.mp4': [],
    '/home/ubuntu/shared/stream/stream_20240521_062836.mp4': [
        [7.23, 7.57]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_063736.mp4': [
        [4.15, 5.07]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_065253.mp4': [
        [0.02, 1.32],
        [2.17, 2.31],
        [3.13, 3.35],
        [3.49, 4.35],
        [4.40, 9.59]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_070254.mp4': [
        [0.00, 9.57]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_071255.mp4': [
        [0.00, 8.38]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_072200.mp4': [
        [0.00, 0.18]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_072737.mp4': [
        [1.22, 1.51]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_073349.mp4': [
        [0.00, 0.38]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_073757.mp4': [
        [0.00, 2.51]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_074121.mp4': [
        [0.00, 0.32]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_074253.mp4': [
        [0.00, 0.44],
        [0.46, 1.38]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_074545.mp4': [
        [0.00, 0.07]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_074708.mp4': [
        [0.00, 1.00]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_080625.mp4': [
        [3.21, 3.30],
        [3.36, 9.58]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_081627.mp4': [
        [0.00, 0.16],
        [0.37, 2.39]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_082443.mp4': [
        [0.00, 0.08]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_090401.mp4': [
        [0.13, 1.13],
        [1.18, 1.31]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_091717.mp4': [
        [0.00, 0.26],
        [0.33, 0.37],
        [2.19, 3.58],
        [4.13, 6.04]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_092713.mp4': [
        [0.00, 1.46],
        [1.57, 2.05],
        [2.25, 13.39]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_093855.mp4': [
        [1.40, 2.20],
        [4.20, 5.10],
        [9.30, 9.39]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_094858.mp4': [
        [0.03, 0.12]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_094958.mp4': [
        [0.00, 1.04],
        [2.33, 3.10],
        [4.20, 4.37]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_095959.mp4': [
        [0.00, 1.35]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_100227.mp4': [
        [0.00, 1.19],
        [1.23, 2.45]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_124902.mp4': [
        [4.43, 7.55]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_131219.mp4': [
        [1.01, 1.38]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_132209.mp4': [
        [0.00, 0.20],
        [0.46, 1.25],
        [1.40, 4.01]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_133201.mp4': [
        [0.29, 1.09]
    ],
    '/home/ubuntu/shared/stream/stream_20240521_132712.mp4': [
        [0.00, 2.52],
        [3.35, 3.41]
    ]
}


def process_videos(videos_dict, capture_fps):
    for video_path, intervals in videos_dict.items():
        print(f"Processing video: {video_path}")
        capture_frames(video_path, intervals, capture_fps)
        
capture_fps = 1  # Number of frames to capture per second

process_videos(videos_dict, capture_fps)
