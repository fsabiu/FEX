#!/usr/bin/env python3
"""
SRT Stream YOLO Inference with KLV Metadata to HLS with ID3 Tags

This script:
1. Reads video + KLV metadata from SRT stream
2. Performs YOLO detection on video frames
3. Combines KLV telemetry with detection results
4. Outputs to HLS stream (via MediaMTX) with ID3 tags containing all metadata

Usage:
    python srt_yolo_hls_inference.py \
        --input-srt 'srt://100.105.188.84:8890' \
        --output-hls 'rtsp://localhost:8554/detected_stream' \
        --model runs/detect/train10/weights/best.pt
"""

import argparse
import cv2
import av
import sys
import struct
import time
import json
import socket
import threading
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SRTYOLOInference")


class KLVDecoder:
    """Decoder for MISB 0601 KLV metadata."""
    
    MISB_0601_KEY = bytes([
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
        0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00
    ])
    
    @staticmethod
    def decode(data):
        """
        Decode MISB 0601 KLV packet.
        
        Args:
            data: Raw KLV packet bytes
            
        Returns:
            Dictionary with decoded telemetry, or None if decoding fails
        """
        try:
            # Check if packet starts with MISB 0601 key
            if not data.startswith(KLVDecoder.MISB_0601_KEY):
                return None
            
            offset = 16  # Skip key
            
            # Parse BER length
            length_byte = data[offset]
            offset += 1
            
            if length_byte < 128:
                value_length = length_byte
            elif length_byte == 0x81:
                value_length = data[offset]
                offset += 1
            elif length_byte == 0x82:
                value_length = struct.unpack('>H', data[offset:offset+2])[0]
                offset += 2
            else:
                return None
            
            # Parse Local Data Set items
            telemetry = {}
            end_offset = offset + value_length
            
            while offset < end_offset and offset < len(data):
                tag = data[offset]
                offset += 1
                
                if offset >= len(data):
                    break
                    
                item_length = data[offset]
                offset += 1
                
                if offset + item_length > len(data):
                    break
                    
                value_bytes = data[offset:offset+item_length]
                offset += item_length
                
                # Decode based on tag
                try:
                    if tag == 2:  # Unix timestamp (microseconds)
                        telemetry['timestamp_us'] = struct.unpack('>Q', value_bytes)[0]
                    elif tag == 13:  # Sensor latitude
                        scaled = struct.unpack('>i', value_bytes)[0]
                        telemetry['latitude'] = scaled / 1e7
                    elif tag == 14:  # Sensor longitude
                        scaled = struct.unpack('>i', value_bytes)[0]
                        telemetry['longitude'] = scaled / 1e7
                    elif tag == 15:  # Sensor true altitude
                        scaled = struct.unpack('>H', value_bytes)[0]
                        telemetry['altitude'] = scaled / 10.0
                    elif tag == 5:  # Platform roll
                        scaled = struct.unpack('>h', value_bytes)[0]
                        telemetry['roll'] = scaled / 100.0
                    elif tag == 6:  # Platform pitch
                        scaled = struct.unpack('>h', value_bytes)[0]
                        telemetry['pitch'] = scaled / 100.0
                    elif tag == 7:  # Platform heading
                        scaled = struct.unpack('>H', value_bytes)[0]
                        telemetry['heading'] = scaled / 100.0
                except struct.error:
                    # Skip malformed fields
                    continue
            
            return telemetry
            
        except Exception as e:
            logger.debug(f"KLV decode error: {e}")
            return None


class SRTYOLOInference:
    """
    Main inference class that reads SRT stream with KLV, performs YOLO detection,
    and outputs to HLS with ID3 metadata tags.
    """
    
    def __init__(self, input_srt, output_rtsp, model_path, conf_threshold=0.25,
                 device='auto', classes=None, show_overlay=True,
                 metadata_file=None, skip_frames=0, srt_latency=120,
                 metadata_host=None, metadata_port=5555):
        """
        Initialize the SRT YOLO inference.
        
        Args:
            input_srt: Input SRT stream URL
            output_rtsp: Output RTSP URL (MediaMTX will convert to HLS)
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on
            classes: List of classes to detect
            show_overlay: Show telemetry overlay on video
            metadata_file: Optional file to save metadata JSON
            skip_frames: Skip N frames between detections (0 = process all)
            srt_latency: SRT latency in milliseconds (default 120ms for low latency)
            metadata_host: Optional host to send metadata via UDP
            metadata_port: UDP port for metadata (default 5555)
        """
        self.input_srt = input_srt
        self.output_rtsp = output_rtsp
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes = classes
        self.show_overlay = show_overlay
        self.metadata_file = metadata_file
        self.skip_frames = skip_frames
        self.srt_latency = srt_latency
        self.metadata_host = metadata_host
        self.metadata_port = metadata_port
        
        self.model = None
        self.container = None
        self.video_writer = None
        self.klv_decoder = KLVDecoder()
        
        self.frame_count = 0
        self.processed_frame_count = 0
        self.klv_count = 0
        self.detection_count = 0
        self.start_time = None
        
        # Store latest KLV data (synchronized with video PTS)
        self.latest_klv = None
        self.klv_pts = None
        
        # Store latest detection results for skipped frames
        self.latest_detections = []
        
        # For metadata export
        self.metadata_buffer = deque(maxlen=1000)  # Keep last 1000 frames of metadata
        
        # UDP socket for metadata streaming
        self.metadata_socket = None
        if self.metadata_host:
            self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"Metadata will be sent to {self.metadata_host}:{self.metadata_port}")
        
        self._stop_event = threading.Event()
    
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
    
    def _create_gstreamer_output(self, width, height, fps):
        """
        Create GStreamer-based VideoWriter for low-latency RTSP output.
        
        This uses OpenCV's VideoWriter with GStreamer backend for better
        performance and lower latency than FFmpeg subprocess.
        """
        # GStreamer pipeline for low-latency RTSP output
        gst_pipeline = (
            'appsrc ! '
            'videoconvert ! '
            'video/x-raw,format=I420 ! '
            'x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast key-int-max=30 ! '
            'video/x-h264,profile=baseline ! '
            'h264parse ! '
            f'rtspclientsink location={self.output_rtsp} protocols=tcp latency=0'
        )
        
        logger.info(f"Creating GStreamer output pipeline:")
        logger.info(f"  Pipeline: {gst_pipeline}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        
        # Create VideoWriter with GStreamer backend
        out = cv2.VideoWriter(
            gst_pipeline,
            cv2.CAP_GSTREAMER,
            0,  # FPS (ignored with GStreamer appsrc)
            fps,
            (width, height),
            True
        )
        
        if not out.isOpened():
            raise RuntimeError(
                "Failed to open GStreamer VideoWriter. "
                "Make sure MediaMTX is running and GStreamer is installed."
            )
        
        logger.info("✓ GStreamer output created successfully")
        return out
    
    def _extract_detections(self, results):
        """
        Extract detection information from YOLO results.
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = results.names.get(class_id, f"class_{class_id}")
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return detections
    
    def _create_metadata_packet(self, klv_data, detections, frame_num, timestamp):
        """
        Create combined metadata packet with KLV telemetry and detections.
        
        Args:
            klv_data: Decoded KLV telemetry dictionary
            detections: List of detection dictionaries
            frame_num: Frame number
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with all metadata
        """
        metadata = {
            'frame': frame_num,
            'timestamp': timestamp,
            'telemetry': klv_data if klv_data else {},
            'detections': detections,
            'detection_count': len(detections)
        }
        
        return metadata
    
    def _overlay_metadata(self, frame, klv_data, detections, fps):
        """
        Overlay telemetry and detection info on frame.
        
        Args:
            frame: Video frame
            klv_data: Decoded KLV telemetry
            detections: Detection results
            fps: Current FPS
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        y_offset = 30
        line_height = 35
        
        # Background for better readability
        cv2.rectangle(overlay, (5, 5), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # FPS and frame count
        cv2.putText(frame, f'FPS: {fps:.1f}', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f'Frame: {self.frame_count}', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
        
        # KLV Telemetry
        if klv_data:
            if 'latitude' in klv_data and 'longitude' in klv_data:
                cv2.putText(frame, f"GPS: {klv_data['latitude']:.6f}, {klv_data['longitude']:.6f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += line_height
            
            if 'altitude' in klv_data:
                cv2.putText(frame, f"Alt: {klv_data['altitude']:.1f}m", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += line_height
            
            if 'heading' in klv_data:
                cv2.putText(frame, f"Heading: {klv_data['heading']:.1f}°", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += line_height
        
        # Detection summary
        if detections:
            det_text = f"Detections: {len(detections)}"
            cv2.putText(frame, det_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += line_height
            
            # Show top 2 detections
            for i, det in enumerate(detections[:2]):
                det_info = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(frame, det_info, 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += line_height
        
        return frame
    
    def start(self):
        """Start the inference process."""
        logger.info(f"Starting SRT YOLO Inference")
        logger.info(f"  Input: {self.input_srt}")
        logger.info(f"  Output: {self.output_rtsp}")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Skip frames: {self.skip_frames}")
        
        try:
            # Load YOLO model
            logger.info("Loading YOLO model...")
            self.model = YOLO(self.model_path)
            self.device = self._resolve_device(self.device)
            logger.info(f"  Device: {self.device}")
            
            # Open SRT stream with low-latency options
            logger.info("Opening SRT stream...")
            srt_options = {
                'timeout': '5000000',  # 5 second timeout
                'recv_buffer_size': '8388608',  # 8MB receive buffer
                'latency': str(self.srt_latency * 1000),  # Convert ms to microseconds
                'payload_size': '1316',  # MTU size
                'max_bw': '0',  # Unlimited bandwidth
                'ffs': '25600',  # Flight flag size
                'ipttl': '64',
                'iptos': '0xB8',  # Low latency ToS
                'tlpktdrop': '1',  # Drop late packets
                'tsbpdmode': '1',  # Timestamp-based packet delivery
            }
            
            logger.info(f"  SRT latency: {self.srt_latency}ms")
            logger.info(f"  Buffer size: 8MB")
            
            self.container = av.open(self.input_srt, options=srt_options)
            
            # Find video and data streams
            video_stream = None
            data_stream = None
            
            for stream in self.container.streams:
                logger.info(f"  Found stream: {stream.type} - {stream}")
                if stream.type == 'video':
                    video_stream = stream
                elif stream.type == 'data':
                    data_stream = stream
            
            if not video_stream:
                logger.error("No video stream found!")
                return False
            
            if not data_stream:
                logger.warning("No data stream (KLV) found - continuing without telemetry")
            else:
                logger.info("✓ KLV metadata stream found")
            
            # Get video properties
            width = video_stream.width
            height = video_stream.height
            fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
            
            logger.info(f"  Video: {width}x{height} @ {fps} fps")
            
            # Create GStreamer output
            logger.info("Creating GStreamer output...")
            self.video_writer = self._create_gstreamer_output(width, height, int(fps))
            
            self.video_stream = video_stream
            self.data_stream = data_stream
            self.fps = fps
            
            logger.info("✓ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Main inference loop."""
        if not self.start():
            return
        
        self.start_time = time.time()
        last_fps_time = self.start_time
        fps_frame_count = 0
        current_fps = 0.0
        
        logger.info("Starting inference loop...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Demux both video and data streams
            streams_to_demux = [self.video_stream]
            if self.data_stream:
                streams_to_demux.append(self.data_stream)
            
            for packet in self.container.demux(streams_to_demux):
                if self._stop_event.is_set():
                    break
                
                # Handle KLV data packets
                if packet.stream.type == 'data':
                    self.klv_count += 1
                    packet_data = bytes(packet)
                    klv_data = self.klv_decoder.decode(packet_data)
                    
                    if klv_data:
                        self.latest_klv = klv_data
                        self.klv_pts = packet.pts
                        logger.debug(f"KLV packet #{self.klv_count}: {klv_data}")
                    
                    continue
                
                # Handle video packets
                if packet.stream.type == 'video':
                    frames = packet.decode()
                    
                    for frame in frames:
                        self.frame_count += 1
                        fps_frame_count += 1
                        
                        # Convert frame to numpy array
                        img = frame.to_ndarray(format='bgr24')
                        
                        # Decide whether to run inference on this frame
                        should_detect = (self.skip_frames == 0) or (self.frame_count % (self.skip_frames + 1) == 1)
                        
                        if should_detect:
                            # Run YOLO inference
                            self.processed_frame_count += 1
                            results = self.model(img, conf=self.conf_threshold, 
                                               verbose=False, device=self.device, 
                                               classes=self.classes)
                            
                            # Get detections
                            detections = self._extract_detections(results[0])
                            if detections:
                                self.detection_count += len(detections)
                            
                            # Store for skipped frames
                            self.latest_detections = detections
                            
                            # Draw YOLO detections
                            annotated_frame = results[0].plot()
                        else:
                            # Use previous detections for skipped frames
                            detections = self.latest_detections
                            annotated_frame = img.copy()
                            
                            # Draw previous detections manually if any
                            if detections:
                                for det in detections:
                                    x1, y1, x2, y2 = map(int, det['bbox'])
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Calculate FPS
                        current_time = time.time()
                        if current_time - last_fps_time >= 1.0:
                            current_fps = fps_frame_count / (current_time - last_fps_time)
                            last_fps_time = current_time
                            fps_frame_count = 0
                        
                        # Add metadata overlay
                        if self.show_overlay:
                            annotated_frame = self._overlay_metadata(
                                annotated_frame, self.latest_klv, detections, current_fps
                            )
                        
                        # Create metadata packet
                        metadata = self._create_metadata_packet(
                            self.latest_klv, detections, self.frame_count,
                            datetime.now().isoformat()
                        )
                        
                        # Store metadata
                        self.metadata_buffer.append(metadata)
                        
                        # Send metadata via UDP if enabled
                        if self.metadata_socket:
                            try:
                                metadata_json = json.dumps(metadata)
                                self.metadata_socket.sendto(
                                    metadata_json.encode('utf-8'),
                                    (self.metadata_host, self.metadata_port)
                                )
                            except Exception as e:
                                logger.debug(f"Error sending metadata via UDP: {e}")
                        
                        # Write frame to GStreamer
                        try:
                            self.video_writer.write(annotated_frame)
                        except Exception as e:
                            logger.error(f"Error writing frame: {e}")
                            break
                        
                        # Log progress
                        if self.frame_count % 100 == 0:
                            logger.info(
                                f"Frames: {self.frame_count} (processed: {self.processed_frame_count}) | "
                                f"KLV packets: {self.klv_count} | "
                                f"Total detections: {self.detection_count} | "
                                f"FPS: {current_fps:.1f}"
                            )
                        
        except KeyboardInterrupt:
            logger.info("Stopping inference...")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the inference."""
        logger.info("Stopping inference...")
        self._stop_event.set()
        
        # Close VideoWriter
        if self.video_writer:
            try:
                self.video_writer.release()
                logger.info("GStreamer output closed")
            except Exception as e:
                logger.error(f"Error closing VideoWriter: {e}")
        
        # Close container
        if self.container:
            try:
                self.container.close()
                logger.info("SRT stream closed")
            except Exception as e:
                logger.error(f"Error closing container: {e}")
        
        # Close metadata socket
        if self.metadata_socket:
            try:
                self.metadata_socket.close()
                logger.info("Metadata socket closed")
            except Exception as e:
                logger.error(f"Error closing metadata socket: {e}")
        
        # Save metadata if requested
        if self.metadata_file and self.metadata_buffer:
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(list(self.metadata_buffer), f, indent=2)
                logger.info(f"✓ Saved metadata to {self.metadata_file}")
            except Exception as e:
                logger.error(f"Error saving metadata: {e}")
        
        # Print summary
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info("\n" + "=" * 70)
        logger.info("📊 Summary:")
        logger.info(f"  Frames processed: {self.frame_count}")
        logger.info(f"  KLV packets: {self.klv_count}")
        logger.info(f"  Total detections: {self.detection_count}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        if self.frame_count > 0:
            logger.info(f"  Average FPS: {self.frame_count / elapsed:.2f}")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SRT Stream YOLO Inference with KLV to HLS with ID3 metadata',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-srt',
        type=str,
        required=True,
        help='Input SRT stream URL (e.g., srt://host:port)'
    )
    parser.add_argument(
        '--output-rtsp',
        type=str,
        default='rtsp://localhost:8554/detected_stream',
        help='Output RTSP URL (MediaMTX will convert to HLS)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='runs/detect/train10/weights/best.pt',
        help='Path to YOLO model'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to run inference on (auto, cpu, 0, 1, etc.)'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        default=None,
        help='List of class IDs to detect'
    )
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Disable telemetry overlay on video'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default=None,
        help='Save metadata to JSON file'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Skip N frames between detections for better performance (0 = process all frames)'
    )
    parser.add_argument(
        '--srt-latency',
        type=int,
        default=120,
        help='SRT latency in milliseconds (lower = less delay, but may cause drops)'
    )
    parser.add_argument(
        '--metadata-host',
        type=str,
        default=None,
        help='Host to send metadata via UDP (e.g., localhost for metadata server)'
    )
    parser.add_argument(
        '--metadata-port',
        type=int,
        default=5555,
        help='UDP port for metadata streaming (default: 5555)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Create and start inference
    inference = SRTYOLOInference(
        input_srt=args.input_srt,
        output_rtsp=args.output_rtsp,
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        classes=args.classes,
        show_overlay=not args.no_overlay,
        metadata_file=args.metadata_file,
        skip_frames=args.skip_frames,
        srt_latency=args.srt_latency,
        metadata_host=args.metadata_host,
        metadata_port=args.metadata_port
    )
    
    try:
        inference.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

