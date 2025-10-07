#!/bin/bash
#
# FEX YOLO Service Wrapper Script
# This script sets up the environment and runs the unified YOLO pipeline
#

# Exit on error
set -e

# Define paths and environment variables
export GST_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
CONDA_BASE="/home/ubuntu/miniforge3"
CONDA_ENV="kuna"
WORK_DIR="/home/ubuntu/FEX"
SCRIPT_NAME="srt_yolo_hls_unified.py"

# Model and stream configuration
INPUT_SRT="srt://100.105.188.84:8890"
OUTPUT_RTSP="rtsp://localhost:8554/detected_stream"
MODEL_PATH="runs/detect/train10/weights/best.pt"
MODE="id3"
SSE_PORT="8081"
METADATA_HOST="127.0.0.1"
METADATA_PORT="5555"

# TAK Server configuration
TAK_ENABLE="true"              # Set to "true" to enable TAK integration
TAK_HOST="localhost"           # TAK Server hostname/IP
TAK_PORT="8089"                # TAK Server SSL port
TAK_CERT="certs/user1.pem"     # Client certificate path
TAK_KEY="certs/user1.key"      # Client key path
TAK_PASSWORD="atakatak"        # Certificate password
TAK_STALE="120"                # Object stale time in seconds (5 minutes)

# Log startup
echo "=========================================="
echo "FEX YOLO Service Starting..."
echo "Date: $(date)"
echo "Working Directory: $WORK_DIR"
echo "Conda Environment: $CONDA_ENV"
echo "=========================================="

# Change to working directory
cd "$WORK_DIR"

# Initialize conda for bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_BASE/etc/profile.d/mamba.sh"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Verify environment is activated
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo "ERROR: Failed to activate conda environment $CONDA_ENV"
    exit 1
fi

echo "Environment activated successfully"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run the application
echo "Starting YOLO pipeline..."
echo "TAK Server Integration: $TAK_ENABLE"

# Build command with TAK parameters if enabled
CMD_ARGS=(
    --input-srt "$INPUT_SRT"
    --output-rtsp "$OUTPUT_RTSP"
    --model "$MODEL_PATH"
    --mode "$MODE"
    --sse-port "$SSE_PORT"
    --metadata-host "$METADATA_HOST"
    --metadata-port "$METADATA_PORT"
    --no-overlay
)

# Add TAK parameters if enabled
if [ "$TAK_ENABLE" = "true" ]; then
    echo "✓ TAK Server enabled: $TAK_HOST:$TAK_PORT"
    CMD_ARGS+=(
        --tak-enable
        --tak-host "$TAK_HOST"
        --tak-port "$TAK_PORT"
        --tak-cert "$TAK_CERT"
        --tak-key "$TAK_KEY"
        --tak-password "$TAK_PASSWORD"
        --tak-stale "$TAK_STALE"
    )
else
    echo "✗ TAK Server disabled"
fi

exec python "$SCRIPT_NAME" "${CMD_ARGS[@]}"

