# Metadata Streaming - Quick Reference

## 🚀 3-Step Setup

### Step 1: Install Dependencies
```bash
pip install websockets
```

### Step 2: Start Everything

**Terminal 1 - MediaMTX:**
```bash
mediamtx mediamtx.yml
```

**Terminal 2 - Inference with Metadata:**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt \
    --metadata-host localhost \
    --srt-latency 100 \
    --skip-frames 1
```

**Terminal 3 - Metadata Server:**
```bash
python3 metadata_server.py
```

**Terminal 4 - Metadata Client:**
```bash
python3 metadata_client.py
```

### Step 3: View Streams

- **Video Stream**: `ffplay http://localhost:8888/detected_stream/index.m3u8`
- **Metadata**: Already showing in Terminal 4

---

## 📺 Expected Output

### Metadata Client Display:
```
Frame #001234 | GPS: 40.712800, -74.006000 | Alt: 152.4m | Hdg: 180.5° | Detections: 2 | [person(0.89), car(0.76)]
Frame #001235 | GPS: 40.712801, -74.006001 | Alt: 152.5m | Hdg: 180.6° | Detections: 1 | [person(0.92)]
Frame #001236 | GPS: 40.712802, -74.006002 | Alt: 152.6m | Hdg: 180.7° | Detections: 3 | [person(0.85), car(0.78), truck(0.65)]
```

---

## 🌍 Remote Client Access

**On Server:**
```bash
python3 metadata_server.py --host 0.0.0.0
```

**On Client (Different Machine):**
```bash
python3 metadata_client.py --server ws://SERVER_IP:8765
```

---

## 💾 Save Metadata to File

```bash
python3 metadata_client.py --output telemetry_log.json
```

---

## 🔍 Verbose Display

```bash
python3 metadata_client.py --verbose
```

Shows full details:
```
================================================================================
Frame #1234 | Timestamp: 2025-10-06T12:34:56
================================================================================

📡 TELEMETRY (KLV):
  GPS: 40.7128000°, -74.0060000°
  Altitude: 152.4 m
  Heading: 180.5°

🎯 DETECTIONS: 2 objects
  1. PERSON (89.00%)
  2. CAR (76.00%)
================================================================================
```

---

## 🔧 Common Commands

| Task | Command |
|------|---------|
| **Start metadata server** | `python3 metadata_server.py` |
| **Connect local client** | `python3 metadata_client.py` |
| **Connect remote client** | `python3 metadata_client.py --server ws://IP:8765` |
| **Save to file** | `python3 metadata_client.py --output data.json` |
| **Verbose output** | `python3 metadata_client.py --verbose` |
| **Check dependencies** | `pip list | grep websockets` |

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't connect | Check metadata_server.py is running |
| No data | Add `--metadata-host localhost` to inference script |
| Wrong port | Match ports: inference `--metadata-port` = server `--udp-port` |

---

## 📦 What You Get

✅ **Real-time GPS coordinates**  
✅ **Altitude, heading, roll, pitch**  
✅ **Object detections with confidence**  
✅ **Bounding box locations**  
✅ **< 20ms latency**  
✅ **Multiple simultaneous clients**  
✅ **JSON export capability**  

---

## 🎯 Architecture

```
Inference Script → UDP (port 5555) → Metadata Server → WebSocket (port 8765) → Clients
```

---

**See METADATA_STREAMING_GUIDE.md for full documentation.**

