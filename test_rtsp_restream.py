import cv2
import time
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Stream processed military detection video to RTSP (outputs from models in models/)')
    parser.add_argument('--video', type=str, 
                       default='datasets/processed_videos/GX010918_military_detected.mp4',
                       help='Path to processed video file')
    parser.add_argument('--output-rtsp', type=str, default='rtsp://localhost:8554/military_detections',
                       help='Output RTSP stream URL')
    parser.add_argument('--fps', type=int, default=25, help='Output FPS for streaming')
    parser.add_argument('--loop', action='store_true', help='Loop the video continuously')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video).exists():
        print(f"Error: Video file {args.video} not found!")
        return
    
    # Open the processed video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Original FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Output FPS: {args.fps}")
    print(f"  - Output RTSP: {args.output_rtsp}")
    print(f"  - Loop: {args.loop}")
    
    # Set up VideoWriter for RTSP output using GStreamer
    gst_str = (
        f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 ! '
        f'h264parse ! rtspclientsink location={args.output_rtsp} protocols=tcp'
    )
    
    out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, args.fps, (width, height), True)
    if not out.isOpened():
        print("Failed to open VideoWriter for RTSP output.")
        print("Make sure you have an RTSP server running (e.g., rtsp-simple-server)")
        cap.release()
        return
    
    print(f"\nStarted streaming military detection video to RTSP...")
    print("Press Ctrl+C to stop.")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    print("Reached end of video, looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Reached end of video.")
                    break
            
            frame_count += 1
            
            # Add frame counter and timestamp to the frame
            current_time = time.time() - start_time
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Time: {current_time:.1f}s', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Write frame to RTSP stream
            out.write(frame)
            
            # Control streaming speed
            time.sleep(1 / args.fps)
            
            # Show progress every 100 frames
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Streamed {frame_count}/{total_frames} frames ({progress:.1f}%)")
                
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        cap.release()
        out.release()
        print("Stream stopped.")

if __name__ == "__main__":
    main()
