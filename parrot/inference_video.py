#!/usr/bin/env python3
"""
YOLOv8 RTSP Inference Script

Runs YOLOv8 inference on an RTSP stream and restreams the output with detections overlaid.

Usage:
    python inference_video.py --input-rtsp rtsp://input_stream --output-rtsp rtsp://output_stream \
        --model runs/detect/train10/weights/best.pt
"""

import argparse
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import time
import threading
import signal
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RTSPInference")


class RTSPInference:
    """
    RTSP inference class that processes input RTSP stream and outputs detected video via RTSP.
    """
    
    def __init__(self, input_rtsp, output_rtsp, model_path, conf_threshold=0.25, 
                 show_fps=True, device='auto', classes=None, buffer_size=10, 
                 save_frames=True, save_interval=10, output_frames_dir="output_frames"):
        """
        Initialize the RTSP inference.
        
        Args:
            input_rtsp: Input RTSP stream URL
            output_rtsp: Output RTSP stream URL
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detections
            show_fps: Whether to show FPS on video
            device: Device to run inference on
            classes: List of classes to detect
            buffer_size: Buffer size for input stream
            save_frames: Whether to save detected frames
            save_interval: Save frame every N frames (default: 10)
            output_frames_dir: Directory to save frames
        """
        self.input_rtsp = input_rtsp
        self.output_rtsp = output_rtsp
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.show_fps = show_fps
        self.device = device
        self.classes = classes
        self.buffer_size = buffer_size
        self.save_frames = save_frames
        self.save_interval = save_interval
        self.output_frames_dir = Path(output_frames_dir)
        
        self.cap = None
        self.out = None
        self.model = None
        self._stop_event = threading.Event()
        self.frame_count = 0
        self.start_time = None
        self.saved_frames_count = 0
        
    def _resolve_device(self, device_value: str) -> str:
        """Resolve device to use for inference."""
        if str(device_value).lower() == 'auto':
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    return '0'  # first GPU
            except ImportError:
                pass
            return 'cpu'
        return str(device_value)
    
    def _save_frame(self, annotated_frame, results):
        """Save annotated frame to output directory."""
        try:
            # Generate filename with timestamp and frame number
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"frame_{self.frame_count:06d}_{timestamp}.jpg"
            filepath = self.output_frames_dir / filename
            
            # Save the frame
            cv2.imwrite(str(filepath), annotated_frame)
            self.saved_frames_count += 1
            
            # Log detection info if any detections found
            if len(results.boxes) > 0:
                detections_info = []
                for box in results.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = results.names.get(class_id, f"class_{class_id}")
                    detections_info.append(f"{class_name}({confidence:.2f})")
                
                logger.info(f"Saved frame {self.frame_count} with {len(results.boxes)} detections: {', '.join(detections_info)}")
            else:
                logger.debug(f"Saved frame {self.frame_count} (no detections)")
                
        except Exception as e:
            logger.error(f"Error saving frame {self.frame_count}: {e}")
    
    def start(self):
        """Start the RTSP inference process."""
        logger.info(f"Starting RTSP inference: {self.input_rtsp} → {self.output_rtsp}")
        
        try:
            # Create output frames directory if saving frames
            if self.save_frames:
                self.output_frames_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output frames directory: {self.output_frames_dir}")
            
            # Load the model
            logger.info(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.device = self._resolve_device(self.device)
            logger.info(f"Using device: {self.device}")
            
            # Open input RTSP stream
            self.cap = cv2.VideoCapture(self.input_rtsp)
            if not self.cap.isOpened():
                logger.error(f"Failed to open input RTSP stream: {self.input_rtsp}")
                return False
            
            # Set buffer size for input stream
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Get input stream properties
            input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Input stream properties:")
            logger.info(f"  - Resolution: {width}x{height}")
            logger.info(f"  - Input FPS: {input_fps}")
            logger.info(f"  - Total frames: {total_frames if total_frames > 0 else 'Live stream'}")
            
            # Set up VideoWriter for RTSP output using GStreamer
            gst_str = (
                f'appsrc ! videoconvert ! '
                f'x264enc tune=zerolatency bitrate=4096 speed-preset=medium ! '
                f'video/x-h264, profile=high ! '
                f'h264parse ! rtspclientsink location={self.output_rtsp} protocols=tcp'
            )
            
            self.out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, input_fps, (width, height), True)
            if not self.out.isOpened():
                logger.error("Failed to open VideoWriter for RTSP output.")
                logger.error("Make sure you have an RTSP server running (e.g., rtsp-simple-server)")
                return False
            
            logger.info(f"Successfully started RTSP inference to: {self.output_rtsp}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting RTSP inference: {e}")
            return False
    
    def run(self):
        """Main inference loop."""
        if not self.start():
            return
        
        self.start_time = time.time()
        logger.info("Press Ctrl+C to stop inference...")
        
        try:
            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from input stream")
                    time.sleep(0.1)  # Brief pause before retrying
                    continue
                
                self.frame_count += 1
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold, verbose=False, 
                                   device=self.device, classes=self.classes)
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
                # Add FPS counter if requested
                if self.show_fps:
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Add FPS text to frame
                    cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add frame counter
                    cv2.putText(annotated_frame, f'Frame: {self.frame_count}', 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add timestamp
                    timestamp = time.strftime('%H:%M:%S', time.localtime())
                    cv2.putText(annotated_frame, f'Time: {timestamp}', 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save frame if enabled and interval reached
                if self.save_frames and self.frame_count % self.save_interval == 0:
                    self._save_frame(annotated_frame, results[0])
                
                # Write frame to output RTSP stream
                self.out.write(annotated_frame)
                
                # Show progress every 100 frames
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames")
                    
        except KeyboardInterrupt:
            logger.info("Stopping inference...")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the inference."""
        logger.info("Stopping RTSP inference...")
        self._stop_event.set()
        
        if self.cap:
            self.cap.release()
            logger.info("Input stream released")
            
        if self.out:
            self.out.release()
            logger.info("Output stream released")
        
        # Log frame saving statistics
        if self.save_frames:
            logger.info(f"✓ Saved {self.saved_frames_count} frames to {self.output_frames_dir}")
        
        logger.info("✓ RTSP inference stopped")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Received interrupt signal, stopping...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on RTSP stream')
    parser.add_argument('--input-rtsp', type=str, required=True,
                       help='Input RTSP stream URL')
    parser.add_argument('--output-rtsp', type=str, 
                       default='rtsp://localhost:8554/detected',
                       help='Output RTSP stream URL')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/train10/weights/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run inference on (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='List of class IDs to detect (e.g., --classes 0 1 2)')
    parser.add_argument('--buffer-size', type=int, default=10,
                       help='Buffer size for input stream (default: 10)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Disable FPS counter overlay')
    parser.add_argument('--save-frames', action='store_true', default=True,
                       help='Save detected frames to output directory (default: True)')
    parser.add_argument('--no-save-frames', action='store_true',
                       help='Disable frame saving')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save frame every N frames (default: 10)')
    parser.add_argument('--output-frames-dir', type=str, default='output_frames',
                       help='Directory to save frames (default: output_frames)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please check the model path and try again")
        sys.exit(1)
    
    # Determine if frame saving should be enabled
    save_frames = args.save_frames and not args.no_save_frames
    
    # Create and start inference
    inference = RTSPInference(
        input_rtsp=args.input_rtsp,
        output_rtsp=args.output_rtsp,
        model_path=args.model,
        conf_threshold=args.conf,
        show_fps=not args.no_fps,
        device=args.device,
        classes=args.classes,
        buffer_size=args.buffer_size,
        save_frames=save_frames,
        save_interval=args.save_interval,
        output_frames_dir=args.output_frames_dir
    )
    
    try:
        inference.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()