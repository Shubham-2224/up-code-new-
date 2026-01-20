# Deployment Guide for EC2 Ubuntu

This directory contains scripts and configurations to run the Voter Extraction application as a systemd service on Ubuntu EC2.

## Prerequisites

1. **Ubuntu LTS** installed on EC2
2. **Python 3.8+** installed
3. **Code uploaded** to EC2 at: `/home/<username>/Projects/OCR PDF TO EXCEL EXTRACTION/PythonData_From_data`
4. **Port 5000** available (or update the service file)

## Quick Setup

### Step 1: Make scripts executable

```bash
cd /home/<username>/Projects/OCR\ PDF\ TO\ EXCEL\ EXTRACTION/PythonData_From_data/deployment
chmod +x setup-service.sh service-manager.sh
```

### Step 2: Run the setup script

```bash
./setup-service.sh
```

This script will:
- Check prerequisites (Python, virtual environment)
- Create virtual environment if needed
- Install dependencies
- Configure and install the systemd service
- Enable the service to start on boot
- Start the service

### Step 3: Verify the service is running

```bash
./service-manager.sh status
```

Or manually:
```bash
sudo systemctl status voter-extraction
```

## Service Management

Use the `service-manager.sh` script for easy management:

```bash
# Start the service
./service-manager.sh start

# Stop the service
./service-manager.sh stop

# Restart the service
./service-manager.sh restart

# Check status
./service-manager.sh status

# View logs (live)
./service-manager.sh logs

# View last 50 lines of logs
./service-manager.sh logs-tail

# Enable auto-start on boot
./service-manager.sh enable

# Disable auto-start on boot
./service-manager.sh disable
```

## Manual Service Commands

You can also use systemctl directly:

```bash
# Check status
sudo systemctl status voter-extraction

# Start service
sudo systemctl start voter-extraction

# Stop service
sudo systemctl stop voter-extraction

# Restart service
sudo systemctl restart voter-extraction

# View logs
sudo journalctl -u voter-extraction -f

# View last 100 lines
sudo journalctl -u voter-extraction -n 100

# Enable on boot
sudo systemctl enable voter-extraction

# Disable on boot
sudo systemctl disable voter-extraction
```

## Accessing the Application

Once the service is running:

- **From EC2 instance**: http://localhost:5000
- **From external**: http://<your-ec2-ip>:5000

**Important**: Make sure your EC2 Security Group allows inbound traffic on port 5000.

## Security Group Configuration

In AWS EC2 Console:
1. Go to Security Groups
2. Select your instance's security group
3. Add inbound rule:
   - Type: Custom TCP
   - Port: 5000
   - Source: 0.0.0.0/0 (or your specific IP for better security)

## Troubleshooting

### Service won't start

1. Check service status:
   ```bash
   sudo systemctl status voter-extraction -l
   ```

2. Check logs:
   ```bash
   sudo journalctl -u voter-extraction -n 50
   ```

3. Verify paths in service file:
   ```bash
   sudo cat /etc/systemd/system/voter-extraction.service
   ```

4. Check if virtual environment exists:
   ```bash
   ls -la /home/<username>/Projects/OCR\ PDF\ TO\ EXCEL\ EXTRACTION/PythonData_From_data/backend/python-service/venv
   ```

### Port 5000 already in use

1. Find what's using the port:
   ```bash
   sudo lsof -i:5000
   ```

2. Kill the process or change the port in `app.py`

### Permission issues

Make sure the service user owns the project directory:
```bash
sudo chown -R $USER:$USER /home/$USER/Projects/
```

### Service file path issues

If your project is in a different location, edit the service file:
```bash
sudo nano /etc/systemd/system/voter-extraction.service
```

Update the `WorkingDirectory` and `ExecStart` paths, then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart voter-extraction
```

## Updating the Application

After updating code:

1. Restart the service:
   ```bash
   sudo systemctl restart voter-extraction
   ```

2. Or use the manager script:
   ```bash
   ./service-manager.sh restart
   ```

## Uninstalling the Service

To remove the service:

```bash
sudo systemctl stop voter-extraction
sudo systemctl disable voter-extraction
sudo rm /etc/systemd/system/voter-extraction.service
sudo systemctl daemon-reload
```

## Files

- `voter-extraction.service` - Systemd service configuration
- `setup-service.sh` - Automated setup script
- `service-manager.sh` - Service management helper
- `README.md` - This file

