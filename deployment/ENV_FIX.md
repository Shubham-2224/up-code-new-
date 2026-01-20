# .env File Issues and Fixes for Ubuntu EC2

## Issues Found in Your .env File

### ❌ **Issue 1: Windows Tesseract Paths**
Your `.env` has Windows paths but you're on Ubuntu:
```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
MARATHI_TRAINEDDATA_PATH=C:\Program Files\Tesseract-OCR\tessdata\mar.traineddata
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

### ✅ **Fix: Use Linux Paths**
```
TESSERACT_CMD=/usr/bin/tesseract
# Note: TESSERACT_CMD is the correct variable name (not TESSERACT_PATH)
# MARATHI_TRAINEDDATA_PATH and TESSDATA_PREFIX are not used by the app
```

### ❌ **Issue 2: GOOGLE_APPLICATION_CREDENTIALS Format**
Currently set to an API key string:
```
GOOGLE_APPLICATION_CREDENTIALS="AIzaSyCYcFZKvk_zmKAyvZFkqmNXTTm-KhQ15Tk"
```

### ✅ **Fix: Should be Path to JSON File**
```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```
OR if you want to use it as an API key (legacy mode), use:
```
GOOGLE_VISION_API_KEY=AIzaSyCYcFZKvk_zmKAyvZFkqmNXTTm-KhQ15Tk
```

### ⚠️ **Issue 3: PORT Setting**
```
PORT=3000
```
This doesn't affect the Flask app (which uses port 5000), but if you have other services, keep it.

## Recommended .env Configuration for Ubuntu EC2

```bash
# ===============================================
# Server Configuration
# ===============================================
DEBUG=False
PORT=5000
HOST=0.0.0.0

# File retention period (hours)
FILE_RETENTION_HOURS=24

# CORS - Allow all origins for EC2 (or specify your domain)
# ALLOWED_ORIGINS=http://your-ec2-ip:5000,http://yourdomain.com:5000

# ===============================================
# Azure Configuration
# ===============================================
AZURE_OPENAI_API_KEY="4gnp3MxigdpSJx9On7iWfsmlysL82533a2Shio8wMa3Ziyzx6jc8JQQJ99BJACfhMk5XJ3w3AAAAACOGTjb2"
AZURE_VISION_ENDPOINT="https://maste-mh4v10u0-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
AZURE_API_KEY="34BHe2UXP2kM05jtnZfuuUhyITQCO1fBTlyUh6nJoLXob5uRmwHsJQQJ99BJACYeBjFXJ3w3AAABACOGvFVO"
OCR_ENDPOINT="https://electionocr.services.ai.azure.com/"
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
OCR_ENGINE=azure_vision

# ===============================================
# Worker Configuration
# ===============================================
POPPLER_MAX_WORKERS=8
AZURE_VISION_MAX_WORKERS=8
AZURE_VISION_PAGE_MAX_WORKERS=2
AZURE_OPENAI_MAX_WORKERS=8
AZURE_OPENAI_FILE_FORMAT_MAX_WORKERS=8
PAIR_SEND_MAX_WORKERS=8
WEBAPP_PAIR_FORMAT_MAX_WORKERS=8

# ===============================================
# Tesseract OCR Configuration (Ubuntu)
# ===============================================
# Tesseract is usually at /usr/bin/tesseract (auto-detected if not set)
TESSERACT_CMD=/usr/bin/tesseract

# OCR Language (Marathi + English)
OCR_LANGUAGE=mar+eng
OCR_OEM=1
OCR_PSM=6
OCR_DPI=400
OCR_CONFIDENCE_THRESHOLD=0.7

# ===============================================
# Google Vision API (Optional)
# ===============================================
# Option 1: Service Account JSON file (Recommended)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option 2: Legacy API Key (if you want to use the key directly)
GOOGLE_VISION_API_KEY=AIzaSyCYcFZKvk_zmKAyvZFkqmNXTTm-KhQ15Tk

# Devanagari Transliterator
devanagari_transliterator=AIzaSyAtG2rmzGISwLa9RUVLq-lyLlv0qTAFUkE

# ===============================================
# Performance Settings
# ===============================================
MAX_BACKGROUND_WORKERS=3
MAX_FILE_SIZE=52428800  # 50MB in bytes

# ===============================================
# Logging
# ===============================================
LOG_LEVEL=INFO
```

## Quick Fix Commands

Run these on your EC2 instance:

```bash
cd /home/ubuntu/ocr/PythonData_From_data/backend/python-service

# Backup current .env
cp .env .env.backup

# Update Tesseract path (if needed)
sed -i 's|TESSERACT_PATH=.*|TESSERACT_CMD=/usr/bin/tesseract|g' .env

# Remove Windows-specific paths (not used by app)
sed -i '/MARATHI_TRAINEDDATA_PATH/d' .env
sed -i '/TESSDATA_PREFIX/d' .env

# Fix GOOGLE_APPLICATION_CREDENTIALS (if using API key, use GOOGLE_VISION_API_KEY instead)
# If you have a JSON file:
# sed -i 's|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json|g' .env

# Or if using API key (legacy):
sed -i 's|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_VISION_API_KEY=AIzaSyCYcFZKvk_zmKAyvZFkqmNXTTm-KhQ15Tk|g' .env
```

## After Making Changes

Restart the service:
```bash
sudo systemctl restart voter-extraction
```

Check if it's working:
```bash
sudo systemctl status voter-extraction
sudo journalctl -u voter-extraction -n 50
```

