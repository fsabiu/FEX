#!/usr/bin/env python3
"""
Test script for SRT YOLO HLS Inference

This script provides example commands and validates the setup.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("=" * 70)
    print("Checking Dependencies")
    print("=" * 70)
    
    checks = {
        'ffmpeg': ['ffmpeg', '-version'],
        'mediamtx': ['which', 'mediamtx'],
        'python_av': ['python3', '-c', 'import av; print(av.__version__)'],
        'ultralytics': ['python3', '-c', 'import ultralytics; print(ultralytics.__version__)'],
        'opencv': ['python3', '-c', 'import cv2; print(cv2.__version__)'],
    }
    
    all_ok = True
    for name, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ {name:15} - OK")
            else:
                print(f"✗ {name:15} - NOT FOUND")
                all_ok = False
        except Exception as e:
            print(f"✗ {name:15} - ERROR: {e}")
            all_ok = False
    
    print()
    return all_ok


def check_model():
    """Check if YOLO model exists."""
    print("=" * 70)
    print("Checking YOLO Model")
    print("=" * 70)
    
    model_path = Path("runs/detect/train10/weights/best.pt")
    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"✗ Model NOT found: {model_path}")
        print("  Please ensure your YOLO model is available")
        return False


def print_usage():
    """Print usage examples."""
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    
    print("\n1. Basic usage (with default model):")
    print("-" * 70)
    print("""
python3 srt_yolo_hls_inference.py \\
    --input-srt 'srt://100.105.188.84:8890' \\
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \\
    --model runs/detect/train10/weights/best.pt
""")
    
    print("\n2. With custom confidence threshold and metadata export:")
    print("-" * 70)
    print("""
python3 srt_yolo_hls_inference.py \\
    --input-srt 'srt://100.105.188.84:8890' \\
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \\
    --model runs/detect/train10/weights/best.pt \\
    --conf 0.5 \\
    --metadata-file metadata.json
""")
    
    print("\n3. With GPU acceleration:")
    print("-" * 70)
    print("""
python3 srt_yolo_hls_inference.py \\
    --input-srt 'srt://100.105.188.84:8890' \\
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \\
    --model runs/detect/train10/weights/best.pt \\
    --device 0
""")
    
    print("\n4. Detect specific classes only:")
    print("-" * 70)
    print("""
python3 srt_yolo_hls_inference.py \\
    --input-srt 'srt://100.105.188.84:8890' \\
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \\
    --model runs/detect/train10/weights/best.pt \\
    --classes 0 1 2
""")


def print_mediamtx_setup():
    """Print MediaMTX setup instructions."""
    print("\n" + "=" * 70)
    print("MediaMTX Setup for HLS Output")
    print("=" * 70)
    
    print("""
1. Install MediaMTX (if not already installed):
   wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_amd64.tar.gz
   tar -xzf mediamtx_v1.5.0_linux_amd64.tar.gz
   sudo mv mediamtx /usr/local/bin/
   sudo chmod +x /usr/local/bin/mediamtx

2. Create MediaMTX configuration file (mediamtx.yml):
""")
    
    config = """
# MediaMTX Configuration for HLS with Metadata
paths:
  detected_stream:
    # Allow publishing from localhost
    publishUser: 
    publishPass: 
    publishIPs: [127.0.0.1]
    
    # HLS Configuration
    runOnDemand: ffmpeg -re -stream_loop -1 -i rtsp://localhost:$RTSP_PORT/$MTX_PATH -c copy -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH
    
# Global HLS settings
hls: yes
hlsAddress: :8888
hlsAlwaysRemux: yes
hlsSegmentCount: 10
hlsSegmentDuration: 1s
hlsPartDuration: 200ms
hlsSegmentMaxSize: 50M

# RTSP settings  
rtspAddress: :8554
protocols: [tcp]
    """
    print(config)
    
    print("""
3. Start MediaMTX:
   mediamtx mediamtx.yml

4. Access HLS stream:
   - HLS URL: http://localhost:8888/detected_stream/index.m3u8
   - Play in VLC, ffplay, or web player
   
5. Test HLS playback:
   ffplay http://localhost:8888/detected_stream/index.m3u8
""")


def print_workflow():
    """Print the complete workflow."""
    print("\n" + "=" * 70)
    print("Complete Workflow")
    print("=" * 70)
    
    print("""
Step 1: Start MediaMTX server
   $ mediamtx mediamtx.yml

Step 2: Start the SRT YOLO inference script
   $ python3 srt_yolo_hls_inference.py \\
       --input-srt 'srt://100.105.188.84:8890' \\
       --output-rtsp 'rtsp://localhost:8554/detected_stream'

Step 3: View the HLS stream
   $ ffplay http://localhost:8888/detected_stream/index.m3u8
   
   Or in a web browser with an HLS player like hls.js

What's happening:
   [SRT Stream] → [YOLO Detection + KLV Extraction] → [RTSP to MediaMTX] → [HLS with ID3 tags]
   
The stream will include:
   - Video with YOLO detection bounding boxes
   - Telemetry overlay (GPS, altitude, heading, etc.)
   - Detection results overlay (object types and confidence)
   - Metadata exported to JSON file (if --metadata-file specified)
""")


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("SRT YOLO HLS Inference - Setup Validator")
    print("=" * 70)
    print()
    
    # Run checks
    deps_ok = check_dependencies()
    model_ok = check_model()
    
    # Print setup instructions
    print_usage()
    print_mediamtx_setup()
    print_workflow()
    
    # Final summary
    print("\n" + "=" * 70)
    print("Setup Status")
    print("=" * 70)
    
    if deps_ok and model_ok:
        print("✓ All checks passed! You're ready to run the inference.")
    else:
        print("⚠ Some checks failed. Please install missing dependencies.")
        if not model_ok:
            print("⚠ YOLO model not found. Please train or download a model first.")
    
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

