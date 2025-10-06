# SRT Stream YOLO Inference with KLV Metadata to HLS

A complete pipeline for processing drone video streams with embedded KLV telemetry, performing AI object detection, and streaming the results as HLS with metadata tags.

## 🎯 Overview

This system provides an end-to-end solution for:

1. **Reading SRT streams** with video and KLV (MISB 0601) metadata
2. **YOLO object detection** on video frames
3. **Combining telemetry + detections** into unified metadata
4. **Streaming to HLS** via MediaMTX for web/mobile playback
5. **ID3 metadata tags** with telemetry and detection results

## 📋 Architecture

```
┌─────────────────┐
│  SRT Stream     │  Video (H.264) + KLV Metadata (MISB 0601)
│  Port 8890      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  srt_yolo_hls_inference.py              │
│  ┌───────────────────────────────────┐  │
│  │ 1. Extract Video Frames           │  │
│  │ 2. Decode KLV Telemetry           │  │
│  │ 3. Run YOLO Detection             │  │
│  │ 4. Combine Metadata               │  │
│  │ 5. Overlay on Video               │  │
│  └───────────────────────────────────┘  │
└────────┬────────────────────────────────┘
         │ RTSP (H.264 + Metadata)
         ▼
┌─────────────────┐
│   MediaMTX      │  Convert RTSP → HLS
│   Port 8554     │  Generate .m3u8 + .ts segments
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HLS Stream     │  http://localhost:8888/detected_stream/index.m3u8
│  Port 8888      │  - Video with detections
└─────────────────┘  - Telemetry overlay
                     - ID3 metadata tags
```

## 📦 Components

### 1. **srt_yolo_hls_inference.py**
Main inference script that:
- Reads SRT stream using PyAV
- Decodes MISB 0601 KLV telemetry (GPS, altitude, attitude, etc.)
- Runs YOLOv8 detection on video frames
- Combines KLV + detection data into metadata packets
- Overlays information on video
- Outputs to RTSP for MediaMTX

### 2. **read_klv_from_stream.py**
Standalone utility to test KLV extraction:
- Reads SRT/RTSP/file streams
- Decodes and displays KLV telemetry
- Useful for debugging metadata issues

### 3. **MediaMTX**
Stream server that:
- Receives RTSP from inference script
- Converts to HLS with low latency
- Serves .m3u8 playlists and .ts segments
- Passes through metadata in MPEG-TS

### 4. **mediamtx.yml**
Configuration for MediaMTX with:
- HLS enabled on port 8888
- RTSP input on port 8554
- Low-latency settings
- CORS enabled for web players

## 🚀 Quick Start

### Prerequisites

```bash
# Python packages
pip install ultralytics opencv-python av numpy

# System packages
sudo apt-get update
sudo apt-get install -y ffmpeg

# MediaMTX
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_amd64.tar.gz
tar -xzf mediamtx_v1.5.0_linux_amd64.tar.gz
sudo mv mediamtx /usr/local/bin/
sudo chmod +x /usr/local/bin/mediamtx
```

### Step 1: Validate Setup

```bash
python3 test_srt_yolo_hls.py
```

This will check:
- ✅ All dependencies installed
- ✅ YOLO model available
- ✅ MediaMTX configured
- 📋 Usage examples

### Step 2: Start MediaMTX

```bash
# Start MediaMTX with the provided config
mediamtx mediamtx.yml
```

You should see:
```
[INF] MediaMTX v1.5.0
[INF] RTSP server listening on :8554
[INF] HLS server listening on :8888
```

### Step 3: Run the Inference

```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt \
    --conf 0.25
```

### Step 4: View the Stream

**Option A: FFplay**
```bash
ffplay http://localhost:8888/detected_stream/index.m3u8
```

**Option B: VLC**
```bash
vlc http://localhost:8888/detected_stream/index.m3u8
```

**Option C: Web Browser**
Use an HLS player like [hls.js](https://github.com/video-dev/hls.js):
```html
<video id="video" controls width="640" height="480"></video>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
  var video = document.getElementById('video');
  var hls = new Hls();
  hls.loadSource('http://YOUR_SERVER_IP:8888/detected_stream/index.m3u8');
  hls.attachMedia(video);
</script>
```

## 🎛️ Command Line Options

### Basic Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-srt` | *required* | SRT stream URL (e.g., `srt://host:port`) |
| `--output-rtsp` | `rtsp://localhost:8554/detected_stream` | Output RTSP URL for MediaMTX |
| `--model` | `runs/detect/train10/weights/best.pt` | Path to YOLO model |
| `--conf` | `0.25` | Detection confidence threshold (0.0-1.0) |
| `--device` | `auto` | Device for inference (`auto`, `cpu`, `0`, `1`) |

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `--classes` | All | Specific class IDs to detect (e.g., `--classes 0 1 2`) |
| `--no-overlay` | False | Disable telemetry overlay on video |
| `--metadata-file` | None | Save metadata to JSON file |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Example Commands

**1. Basic usage:**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream'
```

**2. High confidence threshold + metadata export:**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --conf 0.5 \
    --metadata-file detections.json
```

**3. GPU acceleration:**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --device 0
```

**4. Detect only specific classes (e.g., person=0, car=2):**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --classes 0 2
```

## 📊 Metadata Format

The system generates metadata combining KLV telemetry and YOLO detections:

```json
{
  "frame": 1234,
  "timestamp": "2025-10-06T12:34:56.123456",
  "telemetry": {
    "timestamp_us": 1696596896123456,
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 152.4,
    "roll": -2.5,
    "pitch": 1.2,
    "heading": 180.5
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "bbox": [120, 150, 180, 280]
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.76,
      "bbox": [300, 200, 450, 320]
    }
  ],
  "detection_count": 2
}
```

## 🎥 Video Overlay

The output video includes overlays showing:

- **FPS**: Current processing rate
- **Frame Number**: Current frame count
- **GPS Coordinates**: Latitude, Longitude from KLV
- **Altitude**: Platform altitude in meters
- **Heading**: Platform heading in degrees
- **Detection Count**: Number of objects detected
- **Top Detections**: Class names and confidence scores
- **Bounding Boxes**: YOLO detection boxes (from model.plot())

## 🔧 Troubleshooting

### No video output
- Check MediaMTX is running: `curl http://localhost:8888`
- Verify RTSP URL is correct
- Check MediaMTX logs for connection errors

### No KLV metadata
- Test with: `python3 read_klv_from_stream.py 'srt://100.105.188.84:8890'`
- Ensure SRT stream includes data stream (`-map 0:d` in source)
- Check KLV packets are MISB 0601 format

### Low FPS
- Use GPU: `--device 0`
- Lower resolution if possible
- Reduce confidence threshold
- Use a smaller YOLO model (e.g., YOLOv8n instead of YOLOv8x)

### HLS playback issues
- Check CORS settings in mediamtx.yml
- Verify firewall allows port 8888
- Try different HLS player (VLC vs ffplay vs web)

### Metadata not in HLS
- ID3 tags require custom HLS player support
- Use `--metadata-file` to export to JSON instead
- Consider WebSocket for real-time metadata streaming

## 📁 File Structure

```
FEX/
├── srt_yolo_hls_inference.py    # Main inference script
├── read_klv_from_stream.py      # KLV extraction utility
├── test_srt_yolo_hls.py         # Setup validator
├── mediamtx.yml                  # MediaMTX configuration
├── README_SRT_YOLO_HLS.md       # This file
└── runs/detect/train10/weights/
    └── best.pt                   # Your YOLO model
```

## 🔗 Related Scripts

- **test_klv_receiver.py**: UDP KLV receiver for testing
- **klv_forwarder.py**: Extract KLV and forward via UDP
- **parrot/inference_video.py**: RTSP-based inference (reference)

## 📖 KLV (MISB 0601) Fields Supported

| Tag | Field | Type | Description |
|-----|-------|------|-------------|
| 2 | Timestamp | uint64 | Unix timestamp in microseconds |
| 5 | Platform Roll | int16 | Roll angle in degrees (scaled) |
| 6 | Platform Pitch | int16 | Pitch angle in degrees (scaled) |
| 7 | Platform Heading | uint16 | True heading in degrees (scaled) |
| 13 | Sensor Latitude | int32 | Latitude in degrees (scaled) |
| 14 | Sensor Longitude | int32 | Longitude in degrees (scaled) |
| 15 | Sensor Altitude | uint16 | True altitude in meters (scaled) |

To add more fields, edit the `KLVDecoder.decode()` method in `srt_yolo_hls_inference.py`.

## 🚀 Performance Optimization

### CPU Optimization
- Use smaller model: YOLOv8n or YOLOv8s
- Reduce input resolution
- Skip frames: process every Nth frame
- Disable overlay: `--no-overlay`

### GPU Optimization
- Use `--device 0` (or `cuda:0`)
- Batch processing (requires code modification)
- Use TensorRT for inference acceleration
- Use FP16 precision

### Network Optimization
- Use TCP for SRT: `srt://host:port?mode=caller`
- Reduce HLS segment size
- Enable low-latency HLS in MediaMTX
- Use local MediaMTX instance

## 🎓 Understanding the Flow

### 1. **SRT Stream Input**
Your ffmpeg command creates an SRT stream:
```bash
ffmpeg -i 'srt://100.105.188.84:8890' \
    -map 0:v -map 0:d \
    -c copy output.ts
```
This stream contains:
- **Stream 0:v**: Video (H.264)
- **Stream 0:d**: Data (KLV metadata)

### 2. **PyAV Demuxing**
The script uses PyAV to demux both streams:
```python
for packet in container.demux([video_stream, data_stream]):
    if packet.stream.type == 'data':
        # Decode KLV
    elif packet.stream.type == 'video':
        # Process video frame
```

### 3. **YOLO Inference**
Each frame is processed:
```python
results = model(frame, conf=0.25)
detections = extract_detections(results)
annotated_frame = results[0].plot()
```

### 4. **Metadata Combination**
KLV telemetry + detections → unified metadata:
```python
metadata = {
    'telemetry': klv_data,
    'detections': detections
}
```

### 5. **RTSP Output**
Processed frames sent to MediaMTX via FFmpeg pipe:
```python
ffmpeg_process.stdin.write(frame.tobytes())
```

### 6. **HLS Conversion**
MediaMTX converts RTSP → HLS:
- Segments video into .ts chunks
- Creates .m3u8 playlist
- Serves via HTTP on port 8888

## 🤝 Contributing

To extend this system:

1. **Add KLV fields**: Edit `KLVDecoder.decode()` with new tag mappings
2. **Custom overlays**: Modify `_overlay_metadata()` method
3. **Metadata export formats**: Add export methods (CSV, database, etc.)
4. **WebSocket streaming**: Add real-time metadata websocket server
5. **Recording**: Implement frame/metadata recording to disk

## 📝 License

This project uses:
- **Ultralytics YOLOv8**: AGPL-3.0 license
- **MediaMTX**: MIT license
- **PyAV**: BSD license

## 🆘 Support

For issues:
1. Check logs with `--log-level DEBUG`
2. Test KLV extraction: `python3 read_klv_from_stream.py`
3. Verify MediaMTX: `curl http://localhost:8888`
4. Review MediaMTX logs for RTSP connection status

## 🎉 Success Criteria

You're successfully running when:
- ✅ MediaMTX shows RTSP connection from inference script
- ✅ HLS stream playable in VLC/ffplay
- ✅ Video shows detection bounding boxes
- ✅ Telemetry overlay displays GPS/altitude/heading
- ✅ Metadata file contains combined telemetry + detections

---

**🎬 Ready to go!** Follow the Quick Start section and you'll be streaming in minutes.

