#!/bin/bash

# Service Management Helper Script
# Quick commands to manage the voter-extraction service

SERVICE_NAME="voter-extraction"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    echo ""
    echo "Voter Extraction Service Manager"
    echo "================================="
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start      - Start the service"
    echo "  stop       - Stop the service"
    echo "  restart    - Restart the service"
    echo "  status     - Show service status"
    echo "  logs       - Show service logs (follow mode)"
    echo "  logs-tail  - Show last 50 lines of logs"
    echo "  enable     - Enable service to start on boot"
    echo "  disable    - Disable service from starting on boot"
    echo "  check      - Check if service is running and port is open"
    echo ""
}

check_status() {
    echo ""
    echo "Service Status:"
    echo "==============="
    sudo systemctl status $SERVICE_NAME --no-pager -l
    echo ""
    echo "Port Check:"
    echo "==========="
    if sudo lsof -i:5000 > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Port 5000 is in use"
        sudo lsof -i:5000
    else
        echo -e "${RED}[ERROR]${NC} Port 5000 is not in use"
    fi
    echo ""
}

case "$1" in
    start)
        echo "Starting $SERVICE_NAME..."
        sudo systemctl start $SERVICE_NAME
        sleep 1
        sudo systemctl status $SERVICE_NAME --no-pager -l | head -n 10
        ;;
    stop)
        echo "Stopping $SERVICE_NAME..."
        sudo systemctl stop $SERVICE_NAME
        sleep 1
        sudo systemctl status $SERVICE_NAME --no-pager -l | head -n 10
        ;;
    restart)
        echo "Restarting $SERVICE_NAME..."
        sudo systemctl restart $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager -l | head -n 10
        ;;
    status)
        check_status
        ;;
    logs)
        echo "Showing logs (Press Ctrl+C to exit)..."
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    logs-tail)
        echo "Last 50 lines of logs:"
        sudo journalctl -u $SERVICE_NAME -n 50 --no-pager
        ;;
    enable)
        echo "Enabling $SERVICE_NAME to start on boot..."
        sudo systemctl enable $SERVICE_NAME
        echo -e "${GREEN}[OK]${NC} Service enabled"
        ;;
    disable)
        echo "Disabling $SERVICE_NAME from starting on boot..."
        sudo systemctl disable $SERVICE_NAME
        echo -e "${YELLOW}[OK]${NC} Service disabled"
        ;;
    check)
        check_status
        ;;
    *)
        show_help
        ;;
esac

