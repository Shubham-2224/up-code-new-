# Quick Start - EC2 Ubuntu Deployment

## One-Time Setup

```bash
cd deployment
chmod +x setup-service.sh service-manager.sh
./setup-service.sh
```

That's it! The service will be installed, configured, and started automatically.

## Daily Use

```bash
# Check if service is running
./service-manager.sh status

# View logs
./service-manager.sh logs

# Restart after code updates
./service-manager.sh restart
```

## Access Your Application

- **Local (from EC2)**: http://localhost:5000
- **External**: http://<your-ec2-ip>:5000

**Don't forget**: Open port 5000 in your EC2 Security Group!

## Common Issues

**Service won't start?**
```bash
sudo journalctl -u voter-extraction -n 50
```

**Port already in use?**
```bash
sudo lsof -i:5000
```

**Need to change port?**
Edit `backend/python-service/app.py` line 1008, then restart:
```bash
./service-manager.sh restart
```

