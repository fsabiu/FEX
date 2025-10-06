# Metadata Streaming Guide

## 🎯 Overview

This guide explains how to stream and receive real-time metadata (KLV telemetry + YOLO detections) from your inference pipeline to client machines.

## 📊 Architecture

```
┌─────────────────────────────┐
│ srt_yolo_hls_inference.py   │  Server (inference machine)
│ - Processes SRT stream      │
│ - Runs YOLO detection       │
│ - Extracts KLV telemetry    │
│ - Sends metadata via UDP    │
└──────────┬──────────────────┘
           │ UDP (port 5555)
           ▼
┌─────────────────────────────┐
│ metadata_server.py          │  Server (same or different machine)
│ - Receives UDP metadata     │
│ - Broadcasts via WebSocket  │
└──────────┬──────────────────┘
           │ WebSocket (port 8765)
           ▼
┌─────────────────────────────┐
│ metadata_client.py          │  Client (anywhere on network)
│ - Connects via WebSocket    │
│ - Displays metadata         │
│ - Saves to JSON (optional)  │
└─────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Server and Client on Same Machine

**Terminal 1: Start MediaMTX**
```bash
mediamtx mediamtx.yml
```

**Terminal 2: Start Inference with Metadata Streaming**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt \
    --metadata-host localhost \
    --metadata-port 5555
```

**Terminal 3: Start Metadata Server**
```bash
python3 metadata_server.py --port 8765 --udp-port 5555
```

**Terminal 4: Start Metadata Client**
```bash
python3 metadata_client.py --server ws://localhost:8765
```

### Option 2: Client on Remote Machine

**Server (Inference Machine):**

Terminal 1:
```bash
mediamtx mediamtx.yml
```

Terminal 2:
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt \
    --metadata-host localhost \
    --metadata-port 5555
```

Terminal 3:
```bash
python3 metadata_server.py --host 0.0.0.0 --port 8765 --udp-port 5555
```

**Client (Remote Machine):**
```bash
python3 metadata_client.py --server ws://SERVER_IP:8765
```

Replace `SERVER_IP` with your server's IP address.

## 📋 Component Details

### 1. **srt_yolo_hls_inference.py** (Updated)

Main inference script with metadata streaming support.

**New Options:**
- `--metadata-host`: Host to send metadata (e.g., `localhost`)
- `--metadata-port`: UDP port for metadata (default: 5555)

**Example:**
```bash
python3 srt_yolo_hls_inference.py \
    --input-srt 'srt://100.105.188.84:8890' \
    --output-rtsp 'rtsp://localhost:8554/detected_stream' \
    --model runs/detect/train10/weights/best.pt \
    --metadata-host localhost \
    --metadata-port 5555 \
    --srt-latency 100 \
    --skip-frames 1
```

### 2. **metadata_server.py**

WebSocket server that receives metadata via UDP and broadcasts to clients.

**Options:**
- `--host`: Server host (default: `0.0.0.0` = all interfaces)
- `--port`: WebSocket port (default: 8765)
- `--udp-port`: UDP port to receive metadata (default: 5555)

**Example:**
```bash
# Local only
python3 metadata_server.py

# Allow remote connections
python3 metadata_server.py --host 0.0.0.0 --port 8765
```

### 3. **metadata_client.py**

WebSocket client that displays real-time metadata.

**Options:**
- `--server`: WebSocket server URL (default: `ws://localhost:8765`)
- `--output`: Save metadata to JSON file (optional)
- `--verbose`: Show full metadata display (default: compact)

**Examples:**
```bash
# Basic usage (compact display)
python3 metadata_client.py

# Verbose display
python3 metadata_client.py --verbose

# Save to file
python3 metadata_client.py --output metadata_log.json

# Connect to remote server
python3 metadata_client.py --server ws://192.168.1.100:8765
```

## 📊 Metadata Format

Each metadata packet contains:

```json
{
  "frame": 1234,
  "timestamp": "2025-10-06T12:34:56.123456",
  "telemetry": {
    "timestamp_us": 1696596896123456,
    "latitude": 40.7128000,
    "longitude": -74.0060000,
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
      "bbox": [120.5, 150.3, 180.2, 280.7]
    }
  ],
  "detection_count": 1
}
```

## 🎨 Client Display Modes

### Compact Mode (Default)

```
Frame #001234 | GPS: 40.712800, -74.006000 | Alt: 152.4m | Hdg: 180.5° | Detections: 2 | [person(0.89), car(0.76)]
```

### Verbose Mode (`--verbose`)

```
================================================================================
Frame #1234 | Timestamp: 2025-10-06T12:34:56.123456
================================================================================

📡 TELEMETRY (KLV):
  Timestamp: 2025-10-06T12:34:56.123456
  GPS: 40.7128000°, -74.0060000°
  Altitude: 152.4 m
  Heading: 180.5°
  Roll: -2.5°
  Pitch: 1.2°

🎯 DETECTIONS: 2 objects
  1. PERSON
     Confidence: 89.00%
     BBox: [120, 150, 180, 281]
  2. CAR
     Confidence: 76.00%
     BBox: [300, 200, 450, 320]
================================================================================
```

## 🔧 Advanced Usage

### Save Metadata to File

```bash
python3 metadata_client.py --output telemetry_log.json
```

This creates a JSON array with all received metadata packets:
```json
[
  { "frame": 1, "timestamp": "...", ... },
  { "frame": 2, "timestamp": "...", ... },
  ...
]
```

### Multiple Clients

You can connect multiple clients to the same server:

```bash
# Client 1: Display only
python3 metadata_client.py

# Client 2: Save to file
python3 metadata_client.py --output log1.json

# Client 3: Verbose display
python3 metadata_client.py --verbose
```

### Custom Ports

Use non-standard ports if needed:

```bash
# Server
python3 metadata_server.py --port 9999 --udp-port 6666

# Inference
python3 srt_yolo_hls_inference.py ... --metadata-host localhost --metadata-port 6666

# Client
python3 metadata_client.py --server ws://localhost:9999
```

## 🌐 Network Configuration

### Firewall Rules

**On Server Machine:**

Allow incoming connections:
```bash
# WebSocket port (for clients)
sudo ufw allow 8765/tcp

# UDP port (for inference script)
sudo ufw allow 5555/udp
```

### Cloud/Remote Server

If running on cloud server (AWS, Azure, etc.):
1. Open port 8765 (TCP) in security group
2. Use public IP or domain name for client connection
3. Consider using SSL/TLS for production (wss://)

## 📈 Performance

### Latency

| Component | Latency |
|-----------|---------|
| Inference → UDP | < 1ms |
| UDP → WebSocket | < 5ms |
| WebSocket → Client | < 10ms |
| **Total** | **< 20ms** |

### Bandwidth

Approximate bandwidth per client:
- Compact metadata: ~1-2 KB/frame
- At 30 FPS: ~30-60 KB/s (~0.5 Mbps)
- Very lightweight!

### Scalability

- **Single server**: 100+ concurrent clients
- **Multiple servers**: Use load balancer for more clients
- **UDP reliability**: Consider TCP alternative for critical applications

## 🐛 Troubleshooting

### Client Can't Connect

**Error:** `Connection refused`

**Solutions:**
1. Check metadata server is running:
   ```bash
   ps aux | grep metadata_server
   ```

2. Check port is listening:
   ```bash
   netstat -tuln | grep 8765
   ```

3. Check firewall:
   ```bash
   sudo ufw status
   ```

4. Verify server address:
   ```bash
   # On server
   ip addr show
   ```

### No Metadata Received

**Issue:** Client connected but no data

**Solutions:**
1. Check inference script has `--metadata-host` set:
   ```bash
   ps aux | grep srt_yolo_hls_inference
   ```

2. Check UDP port matches:
   - Inference: `--metadata-port 5555`
   - Server: `--udp-port 5555`

3. Test UDP locally:
   ```bash
   # Send test packet
   echo '{"test":"data"}' | nc -u localhost 5555
   ```

### Metadata Delayed

**Issue:** Metadata arrives late

**Solutions:**
1. Check network latency:
   ```bash
   ping SERVER_IP
   ```

2. Reduce frame processing:
   ```bash
   # Inference script
   --skip-frames 1
   ```

3. Use local metadata server (same machine as inference)

## 💡 Best Practices

### Production Deployment

1. **Use systemd services** for auto-restart:
   ```bash
   # /etc/systemd/system/metadata-server.service
   [Unit]
   Description=Metadata WebSocket Server
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/FEX
   ExecStart=/usr/bin/python3 /home/ubuntu/FEX/metadata_server.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

2. **Use reverse proxy** (nginx) for SSL:
   ```nginx
   location /metadata {
       proxy_pass http://localhost:8765;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
   }
   ```

3. **Monitor connections**:
   ```bash
   # Watch server logs
   python3 metadata_server.py 2>&1 | tee metadata_server.log
   ```

### Development

1. **Test with dummy data**:
   ```python
   import socket
   import json
   
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   metadata = {"frame": 1, "test": "data"}
   sock.sendto(json.dumps(metadata).encode(), ('localhost', 5555))
   ```

2. **Use verbose logging**:
   ```bash
   python3 metadata_client.py --log-level DEBUG
   ```

## 🎓 Integration Examples

### Web Dashboard

```javascript
// Connect to metadata server
const ws = new WebSocket('ws://SERVER_IP:8765');

ws.onmessage = (event) => {
    const metadata = JSON.parse(event.data);
    
    // Update UI
    document.getElementById('gps').textContent = 
        `${metadata.telemetry.latitude}, ${metadata.telemetry.longitude}`;
    
    document.getElementById('detections').textContent = 
        metadata.detection_count;
};
```

### Python Analysis

```python
import asyncio
import websockets
import json

async def analyze_metadata():
    async with websockets.connect('ws://localhost:8765') as ws:
        async for message in ws:
            metadata = json.loads(message)
            
            # Your analysis code
            if metadata['detection_count'] > 5:
                print("High activity detected!")
```

### MQTT Bridge

```python
import asyncio
import websockets
import paho.mqtt.client as mqtt

mqtt_client = mqtt.Client()
mqtt_client.connect("mqtt_broker", 1883)

async def bridge():
    async with websockets.connect('ws://localhost:8765') as ws:
        async for message in ws:
            # Forward to MQTT
            mqtt_client.publish("drone/telemetry", message)
```

## 📚 Summary

| Component | Purpose | Port | Protocol |
|-----------|---------|------|----------|
| **inference script** | Generate metadata | - | - |
| **UDP sender** | Send to server | 5555 | UDP |
| **metadata_server.py** | Broadcast metadata | 8765 | WebSocket |
| **metadata_client.py** | Display/save metadata | - | WebSocket |

---

**🎉 You now have real-time metadata streaming!**

For video stream: `http://SERVER_IP:8888/detected_stream/index.m3u8`  
For metadata: `ws://SERVER_IP:8765`

