#!/usr/bin/env python3
"""
RTSP Restreamer - Forward input RTSP stream to output RTSP stream
Combines functionality from test_rtsp_restream.py and video.py
"""

import cv2
import time
import argparse
import logging
import threading
import signal
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RTSPRestreamer")


class RTSPRestreamer:
    """
    RTSP restreamer that takes an input RTSP stream and forwards it to an output RTSP stream.
    """
    
    def __init__(self, input_rtsp, output_rtsp, fps=None, loop=False, buffer_size=10):
        """
        Initialize the RTSP restreamer.
        
        Args:
            input_rtsp: Input RTSP stream URL
            output_rtsp: Output RTSP stream URL
            fps: Output FPS (if None, uses input FPS)
            loop: Whether to loop the stream
            buffer_size: Buffer size for input stream
        """
        self.input_rtsp = input_rtsp
        self.output_rtsp = output_rtsp
        self.fps = fps
        self.loop = loop
        self.buffer_size = buffer_size
        self.cap = None
        self.out = None
        self._stop_event = threading.Event()
        self.frame_count = 0
        self.start_time = None
        
    def start(self):
        """Start the restreaming process."""
        logger.info(f"Starting RTSP restreamer: {self.input_rtsp} → {self.output_rtsp}")
        
        try:
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
            
            # Use input FPS if not specified
            if self.fps is None:
                self.fps = input_fps
                
            logger.info(f"Input stream properties:")
            logger.info(f"  - Resolution: {width}x{height}")
            logger.info(f"  - Input FPS: {input_fps}")
            logger.info(f"  - Output FPS: {self.fps}")
            logger.info(f"  - Total frames: {total_frames if total_frames > 0 else 'Live stream'}")
            logger.info(f"  - Loop: {self.loop}")
            
            # Set up VideoWriter for RTSP output using GStreamer
            gst_str = (
                f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 ! '
                f'h264parse ! rtspclientsink location={self.output_rtsp} protocols=tcp'
            )
            
            self.out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, self.fps, (width, height), True)
            if not self.out.isOpened():
                logger.error("Failed to open VideoWriter for RTSP output.")
                logger.error("Make sure you have an RTSP server running (e.g., rtsp-simple-server)")
                return False
            
            logger.info(f"Successfully started restreaming to: {self.output_rtsp}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting restreamer: {e}")
            return False
    
    def run(self):
        """Main restreaming loop."""
        if not self.start():
            return
            
        self.start_time = time.time()
        logger.info("Press Ctrl+C to stop restreaming...")
        
        try:
            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    if self.loop:
                        logger.info("Reached end of stream, looping...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.info("Reached end of stream.")
                        break
                
                self.frame_count += 1
                
                # Add frame counter and timestamp to the frame
                current_time = time.time() - self.start_time
                cv2.putText(frame, f'Frame: {self.frame_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Time: {current_time:.1f}s', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Input: {self.input_rtsp.split("/")[-1]}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Write frame to output RTSP stream
                self.out.write(frame)
                
                # Control streaming speed
                time.sleep(1 / self.fps)
                
                # Show progress every 100 frames
                if self.frame_count % 100 == 0:
                    logger.info(f"Restreamed {self.frame_count} frames")
                    
        except KeyboardInterrupt:
            logger.info("Stopping restreamer...")
        except Exception as e:
            logger.error(f"Error during restreaming: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the restreamer."""
        logger.info("Stopping RTSP restreamer...")
        self._stop_event.set()
        
        if self.cap:
            self.cap.release()
            logger.info("Input stream released")
            
        if self.out:
            self.out.release()
            logger.info("Output stream released")
        
        logger.info("✓ RTSP restreamer stopped")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Received interrupt signal, stopping...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Restream input RTSP stream to output RTSP stream')
    parser.add_argument('--input-rtsp', type=str, required=True,
                       help='Input RTSP stream URL')
    parser.add_argument('--output-rtsp', type=str, 
                       default='rtsp://localhost:8554/restreamed',
                       help='Output RTSP stream URL')
    parser.add_argument('--fps', type=int, default=None, 
                       help='Output FPS (if not specified, uses input FPS)')
    parser.add_argument('--loop', action='store_true', 
                       help='Loop the input stream continuously')
    parser.add_argument('--buffer-size', type=int, default=10,
                       help='Buffer size for input stream (default: 10)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start restreamer
    restreamer = RTSPRestreamer(
        input_rtsp=args.input_rtsp,
        output_rtsp=args.output_rtsp,
        fps=args.fps,
        loop=args.loop,
        buffer_size=args.buffer_size
    )
    
    try:
        restreamer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


