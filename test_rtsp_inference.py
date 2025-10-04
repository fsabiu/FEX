#!/usr/bin/env python3
"""
Test script for RTSP inference
"""

import subprocess
import sys
from pathlib import Path

def test_rtsp_inference():
    """Test the RTSP inference script with example parameters."""
    
    # Check if model exists
    model_path = Path("runs/detect/train10/weights/best.pt")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model file exists before running inference")
        return False
    
    # Example command
    cmd = [
        "python", "parrot/inference_video.py",
        "--input-rtsp", "rtsp://localhost:8554/mystream",
        "--output-rtsp", "rtsp://localhost:8554/detected_video",
        "--model", str(model_path),
        "--conf", "0.25",
        "--device", "auto",
        "--save-frames",
        "--save-interval", "10",
        "--output-frames-dir", "output_frames",
        "--log-level", "INFO"
    ]
    
    print("Example command to run RTSP inference:")
    print(" ".join(cmd))
    print("\nTo run with your actual RTSP stream:")
    print("1. Replace 'rtsp://your_input_stream_url' with your actual input RTSP URL")
    print("2. Make sure you have an RTSP server running for output (e.g., rtsp-simple-server)")
    print("3. Ensure the model file exists at the specified path")
    print("4. Run the command")
    
    return True

if __name__ == "__main__":
    test_rtsp_inference()
