import os
import subprocess
from PIL import Image
from io import BytesIO
import time
import sys 

# Custom sorting function
def sort_by_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def build_streaming_command(frame_rate, rtsp_url):
    ffmpeg_cmd = [
            'ffmpeg', 
            # '-y',  # Overwrite output file if it exists
            # '-f', 'rawvideo',    # '-c:v', 'jpg',  # Set codec to PNG
            # '-s', f'1280x720',
            '-framerate', f'{frame_rate}',  # Set frame rate (adjust as needed)
            '-i', '-',  # Read input from stdin
            '-r', '100',
            # '-re',
            # "-af", "atempo=0.5",
            # '-pix_fmt', 'yuv420p',
            # '-bufsize', '64M',
            # '-maxrate', '4M',
            '-c:v', 'libx264',  # Video codec
            '-r', f'{frame_rate}',
            '-preset', 'veryslow',  # Preset for faster encoding
            '-tune', 'zerolatency',  # Tune for streaming
            '-f', 'rtsp',  # Output format is RTSP
            '-rtsp_transport', 'tcp',
            f'{rtsp_url}'  # RTSP stream URL
        ]
    return ffmpeg_cmd

def stream_images_as_rtsp(ffmpeg_process, folder_path):

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files using the custom sorting function
    image_files.sort(key=sort_by_number)
    
    # Stream images to ffmpeg
    for image_file in image_files:

        # Open and convert image to pillow format
        image_path = os.path.join(folder_path, image_file)
        pillow_image = Image.open(image_path)
        
        # Convert pillow image to bytes
        with BytesIO() as bio:
            pillow_image.save(bio, format='JPEG')
            image_bytes = bio.getvalue()
        
        ffmpeg_process.stdin.write(image_bytes)
        
    # Close ffmpeg stdin and wait for process to finish
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    

# Example usage:
def main():
    
    if len(sys.argv) < 4:
        print("Usage: python script.py <folder_path> <rtsp_url> <frame_rate>")
        return
    
    # folder_path = "frames/frame_20240517_101530"
    folder_path = sys.argv[1]
    
    # rtsp_url="rtsp://localhost:8554/mystream"
    rtsp_url = sys.argv[2]

    # frame_rate=8
    frame_rate = sys.argv[3]
    
    # Build ffmpeg command
    ffmpeg_cmd = build_streaming_command(frame_rate,rtsp_url)
    # Start ffmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    stream_images_as_rtsp(ffmpeg_process,folder_path)

    print("RTSP streaming finished.")

if __name__ == "__main__":
    main()
