# Frontend Connection Fix

## Problem
The frontend was hardcoded to connect to `localhost:5000`, which doesn't work when accessing the application from outside the EC2 instance.

## Solution Applied

### Frontend Changes
- Updated `frontend/js/api.js` to automatically detect the current hostname
- Updated `frontend/js/azureVisionIntegration.js` to use dynamic hostname
- The frontend now uses:
  - `localhost:5000` when accessed from localhost (for local development)
  - `<ec2-ip>:5000` or `<domain>:5000` when accessed from EC2 IP/domain

### Backend Changes
- Updated CORS configuration in `backend/python-service/app.py` to allow requests from any origin
- This enables the frontend to connect from any IP/domain

## How It Works

The frontend automatically detects the hostname from `window.location.hostname`:
- If you access `http://localhost:5000` → uses `localhost:5000` for API
- If you access `http://54.123.45.67:5000` → uses `54.123.45.67:5000` for API
- If you access `http://yourdomain.com:5000` → uses `yourdomain.com:5000` for API

## Testing

1. **From EC2 instance:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **From your local machine:**
   - Open browser to: `http://<your-ec2-ip>:5000`
   - The frontend should now connect successfully

3. **Check browser console:**
   - Open Developer Tools (F12)
   - Check Console tab - should see successful health check
   - No more "ERR_BLOCKED_BY_CLIENT" errors

## Security Note

The CORS is currently set to allow all origins (`*`). For better security in production:

1. Set `ALLOWED_ORIGINS` environment variable:
   ```bash
   # In your .env file or systemd service
   ALLOWED_ORIGINS=http://your-ec2-ip:5000,http://yourdomain.com:5000
   ```

2. Or edit the service file:
   ```bash
   sudo nano /etc/systemd/system/voter-extraction.service
   ```
   Add:
   ```
   Environment="ALLOWED_ORIGINS=http://your-ec2-ip:5000,http://yourdomain.com:5000"
   ```

3. Restart the service:
   ```bash
   sudo systemctl restart voter-extraction
   ```

