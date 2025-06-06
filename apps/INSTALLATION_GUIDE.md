# ORCNN Application Installation Guide

This guide provides step-by-step instructions to set up the ORCNN (Oriented R-CNN) application environment from scratch.

## Prerequisites

- Linux system with CUDA support (tested on Linux 6.8.0-1021-oracle)
- Anaconda or Miniconda installed
- CUDA 11.5+ compatible GPU (optional but recommended)
- SSL certificates in `cert/` directory (cert.pem and ck.pem)

## Installation Steps

### 1. Create Conda Environment

Create a new conda environment with Python 3.9:

```bash
conda create -n orcnn-env python=3.9 -y
```

### 2. Activate Environment

Activate the newly created environment:

```bash
conda activate orcnn-env
```

### 3. Install PyTorch with CUDA Support

Install PyTorch compatible with CUDA 11.6:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 4. Install OpenMMLab Dependencies

Install OpenMIM package manager:

```bash
pip install -U openmim
```

Install OpenMMLab packages using MIM:

```bash
mim install mmcv-full==1.7.2
mim install mmdet==2.28.2
mim install mmrotate==0.3.4
```

### 5. Install Application Requirements

Install all remaining dependencies:

```bash
pip install -r requirements_orcnn.txt
```

### 6. Fix NumPy Compatibility (Important!)

The OpenMMLab packages require NumPy 1.x. If you encounter NumPy 2.x compatibility issues, downgrade:

```bash
pip install "numpy<2"
```

### 7. Verify Installation

Test if all packages are installed correctly:

```bash
python -c "import mmrotate, mmdet, mmcv, torch; print('All packages imported successfully')"
```

### 8. Setup SSL Certificates

Ensure you have SSL certificates in the `cert/` directory:
- `cert/cert.pem` - SSL certificate
- `cert/ck.pem` - SSL private key

### 9. Create Logs Directory

Create a directory for application logs:

```bash
mkdir -p logs
```

### 10. Run the Application

Run the application in the background with logging:

```bash
conda activate orcnn-env
python app_orcnn.py > logs/orcnn_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   - Error: CUDA version mismatch between PyTorch and system CUDA
   - Solution: Install PyTorch version compatible with your CUDA version

2. **NumPy Compatibility Issues**
   - Error: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2"
   - Solution: Downgrade NumPy with `pip install "numpy<2"`

3. **Port Already in Use**
   - Error: "Address already in use, Port 2053 is in use"
   - Solution: Kill existing processes with `pkill -f app_orcnn.py` or `kill -9 <PID>`

4. **SSL Certificate Issues**
   - Error: SSL context files not found
   - Solution: Ensure `cert.pem` and `ck.pem` exist in the `cert/` directory

5. **Model Download Issues**
   - The application automatically downloads the oriented R-CNN model on first run
   - Ensure stable internet connection during first startup

### Checking Application Status

1. **Check if app is running:**
   ```bash
   ps aux | grep app_orcnn | grep -v grep
   ```

2. **Check port usage:**
   ```bash
   lsof -ti:2053
   ```

3. **View logs:**
   ```bash
   tail -f logs/orcnn_*.log
   ```

### Environment Verification

To verify your environment is correctly set up:

```bash
conda activate orcnn-env
python -c "
import torch
import mmcv
import mmdet
import mmrotate
import flask
import numpy as np
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy version: {np.__version__}')
print('Environment setup successful!')
"
```

## Application Endpoints

Once running, the application provides the following API:

- **POST /detect_orcnn**: Object detection endpoint
  - Requires: `imageData` (base64 encoded image)
  - Optional: `confidence` threshold, `filter` parameters
  - Returns: JSON with detected objects and bounding boxes

## Performance Notes

- First startup may take longer due to model download and initialization
- GPU acceleration is automatically used if available
- The application runs on HTTPS (port 2053) with SSL certificates
- Debug mode is enabled by default for development

## Version Information

- Python: 3.9
- PyTorch: 1.13.1+cu116
- MMCV: 1.7.2
- MMDetection: 2.28.2
- MMRotate: 0.3.4
- NumPy: 1.26.4

This setup has been tested on Linux 6.8.0-1021-oracle with CUDA 11.5+ support. 