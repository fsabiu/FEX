#!/usr/bin/env python3
"""
Unified SRT → YOLO → RTSP/HLS pipeline with optional ID3v2 injection and SSE/UDP metadata.

Modes:
- auto: Use ID3 (GStreamer GI) when available, else basic pipeline.
- id3:  Require ID3 (error if GI not available).
- basic: Always use basic pipeline (no GI).

Metadata:
- Embeds ID3v2 timed metadata when in id3 mode (via GStreamer id3v2mux).
- Always exposes metadata via optional UDP and optional HTTP SSE for consumers.

Example:
  python3 srt_yolo_hls_unified.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt \
    --mode auto --sse-port 8081 --metadata-host 127.0.0.1 --metadata-port 5555
"""

import argparse
import sys
import struct
import time
import json
import socket
import threading
import logging
from pathlib import Path
from datetime import datetime
from collections import deque

import av
import cv2
import numpy as np
from ultralytics import YOLO


# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SRTYOLOUnified")


def _try_import_gi():
    """Try to import GStreamer GI; return (available: bool, gi, Gst, GstApp)."""
    try:
        import gi  # type: ignore
        gi.require_version('Gst', '1.0')
        gi.require_version('GstApp', '1.0')
        from gi.repository import Gst, GstApp, GLib  # type: ignore
        Gst.init(None)
        try:
            logger.info(f"Using Python: {sys.executable}")
        except Exception:
            pass
        return True, gi, Gst, GstApp
    except Exception as e:
        logger.debug(f"GStreamer GI not available: {e}")
        return False, None, None, None


class KLVDecoder:
    """Decoder for MISB 0601 KLV metadata."""

    MISB_0601_KEY = bytes([
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
        0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00
    ])

    @staticmethod
    def decode(data):
        """Decode MISB 0601 KLV packet to a dict; return None if not applicable."""
        try:
            if not data.startswith(KLVDecoder.MISB_0601_KEY):
                return None

            offset = 16
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

                try:
                    if tag == 2:
                        # Unix Timestamp (microseconds, 8-byte unsigned int)
                        if item_length == 8:
                            telemetry['timestamp_us'] = struct.unpack('>Q', value_bytes)[0]
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 2")
                    elif tag == 5:
                        # Platform Roll (degrees, 2-byte signed int scaled by 100)
                        if item_length == 2:
                            scaled = struct.unpack('>h', value_bytes)[0]
                            telemetry['roll'] = scaled / 100.0
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 5")
                    elif tag == 6:
                        # Platform Pitch (degrees, 2-byte signed int scaled by 100)
                        if item_length == 2:
                            scaled = struct.unpack('>h', value_bytes)[0]
                            telemetry['pitch'] = scaled / 100.0
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 6")
                    elif tag == 7:
                        # Platform Heading (degrees, 2-byte unsigned int scaled by 100)
                        if item_length == 2:
                            scaled = struct.unpack('>H', value_bytes)[0]
                            telemetry['heading'] = scaled / 100.0
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 7")
                    elif tag == 13:
                        # Sensor Latitude (degrees, 4-byte signed int scaled by 1e7)
                        if item_length == 4:
                            scaled = struct.unpack('>i', value_bytes)[0]
                            telemetry['latitude'] = scaled / 1e7
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 13")
                    elif tag == 14:
                        # Sensor Longitude (degrees, 4-byte signed int scaled by 1e7)
                        if item_length == 4:
                            scaled = struct.unpack('>i', value_bytes)[0]
                            telemetry['longitude'] = scaled / 1e7
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 14")
                    elif tag == 15:
                        # Sensor Altitude (meters, 2-byte unsigned int scaled by 10)
                        if item_length == 2:
                            scaled = struct.unpack('>H', value_bytes)[0]
                            telemetry['altitude'] = scaled / 10.0
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 15")
                    elif tag == 18:
                        # Sensor Horizontal Field of View (degrees, 2-byte unsigned int scaled by 100)
                        if item_length == 2:
                            scaled = struct.unpack('>H', value_bytes)[0]
                            telemetry['sensor_h_fov'] = scaled / 100.0
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 18")
                    elif tag == 19:
                        # Sensor Vertical Field of View (degrees, 2-byte unsigned int scaled by 100)
                        if item_length == 2:
                            scaled = struct.unpack('>H', value_bytes)[0]
                            telemetry['sensor_v_fov'] = scaled / 100.0
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 19")
                    elif tag == 21:
                        # Gimbal Roll / Sensor Relative Roll (degrees, 4-byte signed int scaled by 1e6)
                        if item_length == 4:
                            scaled = struct.unpack('>i', value_bytes)[0]
                            telemetry['gimbal_roll'] = scaled / 1e6
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 21")
                    elif tag == 22:
                        # Gimbal Pitch / Sensor Relative Pitch (degrees, 4-byte signed int scaled by 1e6)
                        if item_length == 4:
                            scaled = struct.unpack('>i', value_bytes)[0]
                            telemetry['gimbal_pitch'] = scaled / 1e6
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 22")
                    elif tag == 23:
                        # Gimbal Yaw / Sensor Relative Yaw (degrees, 4-byte signed int scaled by 1e6)
                        if item_length == 4:
                            scaled = struct.unpack('>i', value_bytes)[0]
                            telemetry['gimbal_yaw'] = scaled / 1e6
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 23")
                    elif tag == 102:
                        # Sensor Width (millimeters, 4-byte float)
                        if item_length == 4:
                            telemetry['sensor_width_mm'] = struct.unpack('>f', value_bytes)[0]
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 102")
                    elif tag == 103:
                        # Sensor Height (millimeters, 4-byte float)
                        if item_length == 4:
                            telemetry['sensor_height_mm'] = struct.unpack('>f', value_bytes)[0]
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 103")
                    elif tag == 104:
                        # Focal Length (millimeters, 4-byte float)
                        if item_length == 4:
                            telemetry['focal_length_mm'] = struct.unpack('>f', value_bytes)[0]
                        else:
                            logger.debug(f"Unexpected length {item_length} for tag 104")
                except struct.error:
                    continue

            return telemetry
        except Exception as e:
            logger.debug(f"KLV decode error: {e}")
            return None


def resolve_device(device_value: str) -> str:
    if str(device_value).lower() == 'auto':
        try:
            import torch  # type: ignore
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return '0'
        except Exception:
            pass
        return 'cpu'
    return str(device_value)


def extract_detections(results):
    detections = []
    if len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = results.names.get(class_id, f"class_{class_id}")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    return detections


def create_metadata_packet(klv_data, detections, frame_num, timestamp):
    return {
        'frame': frame_num,
        'timestamp': timestamp,
        'telemetry': klv_data if klv_data else {},
        'detections': detections,
        'detection_count': len(detections)
    }


def overlay_metadata(frame, frame_count, klv_data, detections, fps):
    overlay = frame.copy()
    y_offset = 30
    line_height = 35
    cv2.rectangle(overlay, (5, 5), (400, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += line_height
    cv2.putText(frame, f'Frame: {frame_count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += line_height
    if klv_data:
        if 'latitude' in klv_data and 'longitude' in klv_data:
            cv2.putText(frame, f"GPS: {klv_data['latitude']:.6f}, {klv_data['longitude']:.6f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
        if 'altitude' in klv_data:
            cv2.putText(frame, f"Alt: {klv_data['altitude']:.1f}m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
        if 'heading' in klv_data:
            cv2.putText(frame, f"Heading: {klv_data['heading']:.1f}°", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
    if detections:
        det_text = f"Detections: {len(detections)}"
        cv2.putText(frame, det_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        for det in detections[:2]:
            det_info = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(frame, det_info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
    return frame


# -------------------- SSE Broadcaster (optional) --------------------
class SSEBroadcaster:
    def __init__(self):
        self._subscribers = []  # list[queue.Queue]
        self._lock = threading.Lock()

    def subscribe(self):
        import queue
        q = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, data: str):
        dead = []
        with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(data)
                except Exception:
                    dead.append(q)
            for q in dead:
                if q in self._subscribers:
                    self._subscribers.remove(q)


def start_sse_server(port: int, broadcaster: SSEBroadcaster, stop_event: threading.Event):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug("SSE: " + fmt % args)
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

        def do_GET(self):
            if self.path != '/events':
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

            q = broadcaster.subscribe()
            try:
                self.wfile.write(b":\n\n")
                self.wfile.flush()
                while not stop_event.is_set():
                    try:
                        data = q.get(timeout=0.5)
                    except Exception:
                        continue
                    payload = f"data: {data}\n\n".encode('utf-8')
                    self.wfile.write(payload)
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                broadcaster.unsubscribe(q)

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)

    def serve():
        logger.info(f"SSE server listening on :{port} at /events")
        while not stop_event.is_set():
            server.handle_request()

    t = threading.Thread(target=serve, name="sse-server", daemon=True)
    t.start()
    return server


# -------------------- Pipelines --------------------
class BasePipeline:
    def __init__(self, input_srt, output_rtsp, model_path, conf_threshold=0.25,
                 device='auto', classes=None, show_overlay=True,
                 metadata_file=None, skip_frames=0, srt_latency=120,
                 metadata_host=None, metadata_port=5555,
                 sse_port=None, id3_interval=30):
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
        self.sse_port = sse_port
        self.id3_interval = id3_interval

        self.model = None
        self.container = None
        self.klv_decoder = KLVDecoder()

        self.frame_count = 0
        self.processed_frame_count = 0
        self.klv_count = 0
        self.detection_count = 0
        self.start_time = None
        self.latest_klv = None
        self.klv_pts = None
        self.latest_detections = []
        self.metadata_buffer = deque(maxlen=1000)

        # UDP socket for metadata streaming
        self.metadata_socket = None
        if self.metadata_host:
            self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"Metadata (UDP) → {self.metadata_host}:{self.metadata_port}")

        # SSE server (optional)
        self._stop_event = threading.Event()
        self.sse_broadcaster = SSEBroadcaster() if self.sse_port else None
        self._sse_server = None
        if self.sse_port:
            self._sse_server = start_sse_server(self.sse_port, self.sse_broadcaster, self._stop_event)

    def _open_srt_container(self):
        logger.info("Opening SRT stream…")
        srt_options = {
            'timeout': '5000000',
            'recv_buffer_size': '8388608',
            'latency': str(self.srt_latency * 1000),
            'payload_size': '1316',
            'max_bw': '0',
            'ffs': '25600',
            'ipttl': '64',
            'iptos': '0xB8',
            'tlpktdrop': '1',
            'tsbpdmode': '1',
        }
        self.container = av.open(self.input_srt, options=srt_options)

    def _load_model(self):
        logger.info("Loading YOLO model…")
        self.model = YOLO(self.model_path)
        self.device = resolve_device(self.device)
        logger.info(f"Device: {self.device}")

    def start_common(self):
        self._load_model()
        self._open_srt_container()

        video_stream = None
        data_stream = None
        for stream in self.container.streams:
            logger.info(f"Found stream: {stream.type} - {stream}")
            if stream.type == 'video':
                video_stream = stream
            elif stream.type == 'data':
                data_stream = stream

        if not video_stream:
            raise RuntimeError("No video stream found in SRT input")

        width = video_stream.width
        height = video_stream.height
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        logger.info(f"Video: {width}x{height} @ {fps} fps")

        self.video_stream = video_stream
        self.data_stream = data_stream
        self.fps = fps
        self.start_time = time.time()
        return width, height, int(fps)

    # Hooks implemented by subclasses
    def start(self):
        raise NotImplementedError

    def write_frame(self, frame):
        raise NotImplementedError

    def inject_metadata(self, metadata):
        pass

    def stop(self):
        self._stop_event.set()
        if self.container:
            try:
                self.container.close()
                logger.info("SRT stream closed")
            except Exception as e:
                logger.error(f"Error closing container: {e}")
        if self.metadata_socket:
            try:
                self.metadata_socket.close()
                logger.info("Metadata UDP socket closed")
            except Exception as e:
                logger.error(f"Error closing UDP socket: {e}")
        if self.metadata_file and self.metadata_buffer:
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(list(self.metadata_buffer), f, indent=2)
                logger.info(f"Saved metadata to {self.metadata_file}")
            except Exception as e:
                logger.error(f"Error saving metadata: {e}")
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info("\n" + "=" * 70)
        logger.info("Summary:")
        logger.info(f"  Frames processed: {self.frame_count}")
        logger.info(f"  KLV packets: {self.klv_count}")
        logger.info(f"  Total detections: {self.detection_count}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        if self.frame_count > 0 and elapsed > 0:
            logger.info(f"  Average FPS: {self.frame_count / elapsed:.2f}")
        logger.info("=" * 70)

    def _reconnect_stream(self, max_retries=3, retry_delay=2):
        """Attempt to reconnect to the SRT stream after an error."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{max_retries}...")
                if self.container:
                    try:
                        self.container.close()
                    except Exception:
                        pass
                time.sleep(retry_delay)
                self._open_srt_container()
                
                # Re-identify streams
                video_stream = None
                data_stream = None
                for stream in self.container.streams:
                    if stream.type == 'video':
                        video_stream = stream
                    elif stream.type == 'data':
                        data_stream = stream
                
                if not video_stream:
                    logger.warning("No video stream found after reconnection")
                    continue
                
                self.video_stream = video_stream
                self.data_stream = data_stream
                logger.info("Successfully reconnected to SRT stream")
                return True
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error("Failed to reconnect after all attempts")
        return False

    # Main loop shared logic
    def run(self):
        if not self.start():
            return
        last_fps_time = self.start_time
        fps_frame_count = 0
        current_fps = 0.0
        seen_keyframe = False
        logger.info("Starting inference loop… (Ctrl+C to stop)")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while not self._stop_event.is_set():
                try:
                    streams_to_demux = [self.video_stream]
                    if self.data_stream:
                        streams_to_demux.append(self.data_stream)

                    for packet in self.container.demux(streams_to_demux):
                        if self._stop_event.is_set():
                            break
                        if packet.stream.type == 'data':
                            self.klv_count += 1
                            packet_data = bytes(packet)
                            klv_data = self.klv_decoder.decode(packet_data)
                            if klv_data:
                                self.latest_klv = klv_data
                                self.klv_pts = packet.pts
                                # Log KLV data every 5 packets for debugging/monitoring
                                if self.klv_count % 5 == 0:
                                    logger.debug(f"KLV data (packet {self.klv_count}): {klv_data}")
                            continue

                        if packet.stream.type == 'video':
                            # Wait for first keyframe to avoid decoder errors when starting mid-stream
                            if not seen_keyframe:
                                try:
                                    if getattr(packet, 'is_keyframe', False):
                                        seen_keyframe = True
                                    else:
                                        continue
                                except Exception:
                                    # If property not available, attempt decode and rely on error handling
                                    pass

                            try:
                                frames = packet.decode()
                            except Exception as dec_err:
                                # Corrupt/truncated packet (common on live joins). Skip and continue
                                logger.debug(f"Decode error skipped: {dec_err}")
                                continue
                            for frame in frames:
                                self.frame_count += 1
                                fps_frame_count += 1

                                try:
                                    img = frame.to_ndarray(format='bgr24')
                                except Exception as conv_err:
                                    logger.debug(f"Frame convert error skipped: {conv_err}")
                                    continue
                                should_detect = (self.skip_frames == 0) or (self.frame_count % (self.skip_frames + 1) == 1)
                                if should_detect:
                                    self.processed_frame_count += 1
                                    results = self.model(img, conf=self.conf_threshold, verbose=False, device=self.device, classes=self.classes)
                                    detections = extract_detections(results[0])
                                    if detections:
                                        self.detection_count += len(detections)
                                    self.latest_detections = detections
                                    annotated_frame = results[0].plot()
                                else:
                                    detections = self.latest_detections
                                    annotated_frame = img.copy()
                                    if detections:
                                        for det in detections:
                                            x1, y1, x2, y2 = map(int, det['bbox'])
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            label = f"{det['class_name']}: {det['confidence']:.2f}"
                                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                now = time.time()
                                if now - last_fps_time >= 1.0:
                                    current_fps = fps_frame_count / (now - last_fps_time)
                                    last_fps_time = now
                                    fps_frame_count = 0

                                if self.show_overlay:
                                    annotated_frame = overlay_metadata(annotated_frame, self.frame_count, self.latest_klv, detections, current_fps)

                                metadata = create_metadata_packet(self.latest_klv, detections, self.frame_count, datetime.now().isoformat())
                                self.metadata_buffer.append(metadata)

                                # UDP
                                if self.metadata_socket:
                                    try:
                                        self.metadata_socket.sendto(json.dumps(metadata).encode('utf-8'), (self.metadata_host, self.metadata_port))
                                    except Exception:
                                        pass

                                # SSE
                                if self.sse_broadcaster:
                                    try:
                                        self.sse_broadcaster.publish(json.dumps(metadata, separators=(',', ':')))
                                    except Exception:
                                        pass

                                # Optional in-band injection (implemented by subclass)
                                self.inject_metadata(metadata)

                                # Write frame
                                self.write_frame(annotated_frame)

                                if self.frame_count % 100 == 0:
                                    logger.info(f"Frames: {self.frame_count} (processed: {self.processed_frame_count}) | KLV: {self.klv_count} | Detections: {self.detection_count} | FPS: {current_fps:.1f}")
                    
                    # If we exit the demux loop normally, break the outer loop
                    break
                    
                except av.error.OSError as av_err:
                    # Handle SRT stream errors (e.g., decoding errors, I/O errors)
                    consecutive_errors += 1
                    logger.warning(f"SRT stream error (attempt {consecutive_errors}/{max_consecutive_errors}): {av_err}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive errors, attempting reconnection")
                        if not self._reconnect_stream():
                            logger.error("Reconnection failed, stopping stream")
                            break
                        # Reset error counter and keyframe flag after successful reconnection
                        consecutive_errors = 0
                        seen_keyframe = False
                    else:
                        # Wait a bit before continuing
                        time.sleep(0.5)
                        continue

        except KeyboardInterrupt:
            logger.info("Stopping…")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()


class BasicPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_writer = None

    def _create_gst_writer(self, width, height, fps):
        gst_pipeline = (
            'appsrc ! '
            'videoconvert ! '
            'video/x-raw,format=I420 ! '
            'x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast key-int-max=30 ! '
            'video/x-h264,profile=baseline ! '
            'h264parse ! '
            f'rtspclientsink location={self.output_rtsp} protocols=tcp latency=0'
        )
        logger.info("Creating GStreamer VideoWriter (basic mode)…")
        logger.info(f"  Pipeline: {gst_pipeline}")
        out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
        if not out.isOpened():
            raise RuntimeError("Failed to open GStreamer VideoWriter. Ensure MediaMTX and GStreamer are installed.")
        return out

    def start(self):
        width, height, fps = self.start_common()
        self.video_writer = self._create_gst_writer(width, height, fps)
        return True

    def write_frame(self, frame):
        try:
            self.video_writer.write(frame)
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            raise

    def stop(self):
        if self.video_writer:
            try:
                self.video_writer.release()
                logger.info("GStreamer VideoWriter closed")
            except Exception as e:
                logger.error(f"Error closing VideoWriter: {e}")
        super().stop()


class ID3Pipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        available, gi, Gst, GstApp = _try_import_gi()
        if not available:
            raise RuntimeError("GStreamer GI not available; cannot run ID3 pipeline")
        self.Gst = Gst
        self.GstApp = GstApp
        self.pipeline = None
        self.appsrc = None
        self.mpegtsmux = None
        self.ffmpeg_process = None
        self.gst_timestamp = 0
        self.frame_duration = 0
        self._id3_counter = 0

    def _create_pipeline(self, width, height, fps):
        Gst = self.Gst
        # Verify required elements exist in this GStreamer installation
        required = ['appsrc', 'videoconvert', 'videoscale', 'x264enc', 'h264parse', 'mpegtsmux', 'fdsink']
        missing = [name for name in required if Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError(
                "Missing GStreamer elements: " + ', '.join(missing) +
                ". Ensure system Python (/usr/bin/python3) and packages: "
                "python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 "
                "gstreamer1.0-plugins-base gstreamer1.0-plugins-good "
                "gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly"
            )

        # Start ffmpeg process to push to MediaMTX via RTSP
        # Using flv format to preserve metadata
        import subprocess
        self.ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-f', 'mpegts',
            '-i', 'pipe:0',
            '-c:v', 'copy',
            '-metadata', 'title=YOLO Detection Stream',
            '-metadata', 'comment=Contains detection and telemetry metadata',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.output_rtsp
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        pipeline = Gst.Pipeline.new("id3-pipeline")

        appsrc = Gst.ElementFactory.make("appsrc", "source")
        if appsrc is None:
            raise RuntimeError("Failed to create 'appsrc'")
        appsrc.set_property("format", self.Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("block", True)
        caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height},framerate={int(fps)}/1")
        appsrc.set_property("caps", caps)

        videoconvert = Gst.ElementFactory.make("videoconvert", "convert")
        videoscale = Gst.ElementFactory.make("videoscale", "scale")
        x264enc = Gst.ElementFactory.make("x264enc", "encoder")
        if x264enc is None:
            raise RuntimeError("Failed to create 'x264enc' (install gstreamer1.0-plugins-ugly)")
        x264enc.set_property("tune", "zerolatency")
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("bitrate", 4000)
        x264enc.set_property("key-int-max", 30)
        x264enc.set_property("threads", 4)

        h264parse = Gst.ElementFactory.make("h264parse", "parser")
        
        # Create mpegtsmux with metadata support
        mpegtsmux = Gst.ElementFactory.make("mpegtsmux", "mux")
        if mpegtsmux is None:
            raise RuntimeError("Failed to create 'mpegtsmux' (install gstreamer1.0-plugins-bad)")
        mpegtsmux.set_property("alignment", 7)
        
        # Use fdsink to pipe to ffmpeg
        fdsink = Gst.ElementFactory.make("fdsink", "sink")
        if fdsink is None:
            raise RuntimeError("Failed to create 'fdsink'")
        fdsink.set_property("fd", self.ffmpeg_process.stdin.fileno())
        fdsink.set_property("sync", False)

        for e in [appsrc, videoconvert, videoscale, x264enc, h264parse, mpegtsmux, fdsink]:
            if not e:
                raise RuntimeError("Failed to create GStreamer element")
            pipeline.add(e)

        if not appsrc.link(videoconvert):
            raise RuntimeError("Failed to link appsrc → videoconvert")
        if not videoconvert.link(videoscale):
            raise RuntimeError("Failed to link videoconvert → videoscale")
        if not videoscale.link(x264enc):
            raise RuntimeError("Failed to link videoscale → x264enc")
        if not x264enc.link(h264parse):
            raise RuntimeError("Failed to link x264enc → h264parse")
        if not h264parse.link(mpegtsmux):
            raise RuntimeError("Failed to link h264parse → mpegtsmux")
        if not mpegtsmux.link(fdsink):
            raise RuntimeError("Failed to link mpegtsmux → fdsink")

        ret = pipeline.set_state(self.Gst.State.PLAYING)
        if ret == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")

        self.pipeline = pipeline
        self.appsrc = appsrc
        self.mpegtsmux = mpegtsmux
        self.frame_duration = int(self.Gst.SECOND / fps)
        self.gst_timestamp = 0
        logger.info("ID3 pipeline started (using MPEG-TS with inline metadata, publishing via ffmpeg)")

    def start(self):
        width, height, fps = self.start_common()
        self._create_pipeline(width, height, fps)
        return True

    def write_frame(self, frame):
        data = frame.tobytes()
        buf = self.Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = self.gst_timestamp
        buf.duration = self.frame_duration
        self.gst_timestamp += self.frame_duration
        ret = self.appsrc.emit("push-buffer", buf)
        if ret != self.Gst.FlowReturn.OK:
            raise RuntimeError(f"Error pushing buffer: {ret}")

    def inject_metadata(self, metadata):
        # Inject every id3_interval frames as custom MPEG-TS metadata
        if self.frame_count % max(1, self.id3_interval) != 0:
            return
        try:
            taglist = self.Gst.TagList.new_empty()
            telemetry = metadata.get('telemetry', {})
            if 'latitude' in telemetry and 'longitude' in telemetry:
                gps_string = f"{telemetry['latitude']:.7f},{telemetry['longitude']:.7f}"
                taglist.add_value(self.Gst.TagMergeMode.APPEND, 'geo-location-name', gps_string)
                taglist.add_value(self.Gst.TagMergeMode.APPEND, 'geo-location-latitude', telemetry['latitude'])
                taglist.add_value(self.Gst.TagMergeMode.APPEND, 'geo-location-longitude', telemetry['longitude'])
            if 'altitude' in telemetry:
                taglist.add_value(self.Gst.TagMergeMode.APPEND, 'geo-location-elevation', telemetry['altitude'])
            if 'detection_count' in metadata:
                taglist.add_value(self.Gst.TagMergeMode.APPEND, 'comment', f"Detections: {metadata['detection_count']}")
            taglist.add_value(self.Gst.TagMergeMode.APPEND, 'extended-comment', json.dumps(metadata, separators=(',', ':')))
            event = self.Gst.Event.new_tag(taglist)
            # Send tag event to mpegtsmux for inline metadata
            self.mpegtsmux.send_event(event)
            self._id3_counter += 1
        except Exception as e:
            logger.error(f"Error injecting MPEG-TS metadata: {e}")

    def stop(self):
        if self.appsrc:
            try:
                self.appsrc.emit("end-of-stream")
                logger.info("Sent EOS to GStreamer pipeline")
            except Exception as e:
                logger.error(f"Error sending EOS: {e}")
        if self.pipeline:
            try:
                time.sleep(0.3)
                self.pipeline.set_state(self.Gst.State.NULL)
                logger.info("GStreamer pipeline stopped")
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
                logger.info("FFmpeg process stopped")
            except Exception as e:
                logger.error(f"Error stopping ffmpeg: {e}")
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
        super().stop()


def build_pipeline(mode: str, **kwargs) -> BasePipeline:
    mode = mode.lower()
    available, _, _, _ = _try_import_gi()
    if mode == 'id3':
        if not available:
            raise RuntimeError("Mode 'id3' requested but GStreamer GI not available")
        return ID3Pipeline(**kwargs)
    if mode == 'auto':
        if available:
            logger.info("GI available → using ID3 pipeline")
            return ID3Pipeline(**kwargs)
        logger.info("GI not available → using basic pipeline")
        return BasicPipeline(**kwargs)
    if mode == 'basic':
        return BasicPipeline(**kwargs)
    raise ValueError("Invalid mode; expected auto|id3|basic")


def main():
    parser = argparse.ArgumentParser(description='SRT → YOLO → RTSP/HLS with optional ID3 and SSE metadata', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-srt', type=str, required=True, help='Input SRT URL (e.g., srt://host:port)')
    parser.add_argument('--output-rtsp', type=str, default='rtsp://localhost:8554/detected_stream', help='Output RTSP URL (MediaMTX will convert to HLS)')
    parser.add_argument('--model', type=str, default='runs/detect/train10/weights/best.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to run inference on (auto, cpu, 0, 1, …)')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='List of class IDs to detect')
    parser.add_argument('--no-overlay', action='store_true', help='Disable overlay on video')
    parser.add_argument('--metadata-file', type=str, default=None, help='Save metadata to JSON file')
    parser.add_argument('--skip-frames', type=int, default=0, help='Skip N frames between detections (0 = all frames)')
    parser.add_argument('--srt-latency', type=int, default=120, help='SRT latency in milliseconds')
    parser.add_argument('--metadata-host', type=str, default=None, help='Host to send metadata via UDP')
    parser.add_argument('--metadata-port', type=int, default=5555, help='UDP port for metadata')
    parser.add_argument('--sse-port', type=int, default=None, help='Start SSE server on this port (path: /events)')
    parser.add_argument('--id3-interval', type=int, default=30, help='Insert ID3 tag every N frames (ID3 mode)')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'id3', 'basic'], help='Pipeline selection mode')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    try:
        pipeline = build_pipeline(
            mode=args.mode,
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
            metadata_port=args.metadata_port,
            sse_port=args.sse_port,
            id3_interval=args.id3_interval,
        )
        pipeline.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


