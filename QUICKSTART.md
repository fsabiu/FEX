# Quick Start Guide - SRT YOLO HLS Pipeline

## 🚀 In 4 Steps

### 1️⃣ Install Dependencies (One Time)
```bash
# Python packages
pip install ultralytics opencv-python av numpy

# Install MediaMTX
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_amd64.tar.gz
tar -xzf mediamtx_v1.5.0_linux_amd64.tar.gz
sudo mv mediamtx /usr/local/bin/
sudo chmod +x /usr/local/bin/mediamtx
```

### 2️⃣ Validate Setup
```bash
python3 test_srt_yolo_hls.py
```

### 3️⃣ Start MediaMTX (Terminal 1)
```bash
mediamtx mediamtx.yml
```

Expected output:
```
[INF] MediaMTX v1.5.0
[INF] RTSP server listening on :8554
[INF] HLS server listening on :8888
```

### 4️⃣ Run Inference (Terminal 2)
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt
```

### 5️⃣ Watch Stream (Terminal 3)
```bash
ffplay http://localhost:8888/detected_stream/index.m3u8
```

---

## 🎯 What You'll See

✅ Video with YOLO detection boxes  
✅ GPS coordinates overlay  
✅ Altitude and heading display  
✅ Detection counts and class names  
✅ Real-time FPS counter  

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Latency | 2-5 seconds (HLS) |
| FPS | 15-30 fps (CPU), 30-60 fps (GPU) |
| KLV Update Rate | ~1 Hz (depends on source) |
| Detection Accuracy | Model-dependent |

---

## 🛠️ Common Commands

### Test KLV Extraction Only
```bash
python3 read_klv_from_stream.py 'srt://100.105.188.84:8890' --duration 10
```

### Export Metadata to JSON
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --metadata-file output_metadata.json
```

### Use GPU Acceleration
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --device 0
```

### Higher Confidence Threshold
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --conf 0.5
```

---

## 🔗 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| HLS Stream | `http://localhost:8888/detected_stream/index.m3u8` | Main video output |
| RTSP Input | `rtsp://localhost:8554/detected_stream` | MediaMTX input |
| MediaMTX API | `http://localhost:9997` | Server status |
| Metrics | `http://localhost:9998/metrics` | Prometheus metrics |

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "No video stream found" | Check SRT URL is correct and streaming |
| "No data stream found" | Source may not have KLV metadata (will work without it) |
| "Model file not found" | Check path to YOLO model: `runs/detect/train10/weights/best.pt` |
| "FFmpeg pipe broken" | MediaMTX not running or wrong RTSP URL |
| Low FPS | Use `--device 0` for GPU or smaller YOLO model |
| HLS not playing | Check MediaMTX is running on port 8888 |

---

## 📁 Files You Need

```
FEX/
├── srt_yolo_hls_inference.py        # ← Main script
├── mediamtx.yml                      # ← MediaMTX config
├── runs/detect/train10/weights/
│   └── best.pt                       # ← Your YOLO model
└── test_srt_yolo_hls.py             # ← Setup validator
```

---

## 🎓 Architecture Flow

```
SRT Stream (Video+KLV)
    ↓
srt_yolo_hls_inference.py
    ├─→ Extract KLV telemetry
    ├─→ Run YOLO detection  
    ├─→ Combine metadata
    └─→ Overlay on video
    ↓
RTSP to MediaMTX (localhost:8554)
    ↓
MediaMTX converts to HLS
    ↓
HLS Stream (localhost:8888)
    ├─→ Web browser (hls.js)
    ├─→ VLC player
    └─→ ffplay
```

---

## 🎬 Ready!

Follow steps 1-5 above and you'll be streaming in minutes!

For detailed documentation, see: **README_SRT_YOLO_HLS.md**

