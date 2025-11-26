#!/bin/bash

# Fly.io Scaling Script for ruvtrade GPU instances
# This script manages scaling and cost optimization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_NAME="ruvtrade"
FLYCTL_PATH="/home/codespace/.fly/bin/flyctl"

# Function to display help
show_help() {
    echo -e "${BLUE}Fly.io GPU Scaling Script${NC}"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  up            Scale up to production configuration"
    echo "  down          Scale down to minimum configuration"
    echo "  stop          Stop all machines"
    echo "  start         Start stopped machines"
    echo "  status        Show current scaling status"
    echo "  auto          Enable auto-scaling"
    echo "  manual        Disable auto-scaling"
    echo "  cost          Show cost optimization suggestions"
    echo
    echo "Options:"
    echo "  --instances N Set number of instances (1-5)"
    echo "  --memory N    Set memory size (8gb, 16gb, 32gb)"
    echo "  --gpu TYPE    Set GPU type (a10, a100-40gb, a100-80gb)"
    echo
    echo "Examples:"
    echo "  $0 up --instances 2"
    echo "  $0 down"
    echo "  $0 status"
    echo "  $0 cost"
}

# Function to check authentication
check_auth() {
    if ! $FLYCTL_PATH auth whoami > /dev/null 2>&1; then
        echo -e "${RED}❌ Not authenticated with Fly.io${NC}"
        echo -e "${BLUE}Please run: $FLYCTL_PATH auth login${NC}"
        exit 1
    fi
}

# Function to show current status
show_status() {
    echo -e "${BLUE}📊 Current Status for $APP_NAME${NC}"
    echo
    $FLYCTL_PATH status --app "$APP_NAME"
    echo
    echo -e "${BLUE}💰 Current machine costs:${NC}"
    $FLYCTL_PATH machine list --app "$APP_NAME" --json | jq -r '.[] | "\(.id): \(.config.size) - \(.region)"'
}

# Function to scale up
scale_up() {
    local instances=${1:-2}
    echo -e "${BLUE}⬆️ Scaling up to $instances instances${NC}"
    
    # Set machine configuration for production
    $FLYCTL_PATH machine update \
        --config fly.toml \
        --app "$APP_NAME" \
        --yes
    
    # Scale to desired number of instances
    $FLYCTL_PATH scale count $instances --app "$APP_NAME"
    
    echo -e "${GREEN}✅ Scaled up to $instances instances${NC}"
}

# Function to scale down
scale_down() {
    echo -e "${YELLOW}⬇️ Scaling down to minimum configuration${NC}"
    
    # Scale to 1 instance
    $FLYCTL_PATH scale count 1 --app "$APP_NAME"
    
    echo -e "${GREEN}✅ Scaled down to 1 instance${NC}"
}

# Function to stop all machines
stop_machines() {
    echo -e "${YELLOW}⏹️ Stopping all machines${NC}"
    $FLYCTL_PATH machine stop --app "$APP_NAME" --all
    echo -e "${GREEN}✅ All machines stopped${NC}"
}

# Function to start machines
start_machines() {
    echo -e "${BLUE}▶️ Starting machines${NC}"
    $FLYCTL_PATH machine start --app "$APP_NAME" --all
    echo -e "${GREEN}✅ All machines started${NC}"
}

# Function to enable auto-scaling
enable_autoscaling() {
    echo -e "${BLUE}🔄 Enabling auto-scaling${NC}"
    
    # Update fly.toml with autoscaling settings
    cat > /tmp/autoscale_config.toml << EOF
[autoscaling]
  enabled = true
  min_replicas = 1
  max_replicas = 3
  target_cpu_percent = 70
  target_memory_percent = 80
  scale_up_cooldown = "5m"
  scale_down_cooldown = "10m"
EOF
    
    echo -e "${GREEN}✅ Auto-scaling configuration ready${NC}"
    echo -e "${YELLOW}Note: Update your fly.toml with the autoscaling config${NC}"
}

# Function to show cost optimization
show_cost_optimization() {
    echo -e "${BLUE}💰 Cost Optimization Report${NC}"
    echo
    echo -e "${YELLOW}Current Configuration:${NC}"
    $FLYCTL_PATH machine list --app "$APP_NAME"
    echo
    echo -e "${YELLOW}💡 Cost Optimization Tips:${NC}"
    echo "1. Use auto-stop for development workloads"
    echo "2. Scale down during low-traffic periods"
    echo "3. Consider A10 GPUs for less intensive workloads"
    echo "4. Use spot instances when available"
    echo "5. Monitor GPU utilization to avoid over-provisioning"
    echo
    echo -e "${YELLOW}🔧 Recommended commands:${NC}"
    echo "• Development: $0 down"
    echo "• Production: $0 up --instances 2"
    echo "• Maintenance: $0 stop"
    echo
    echo -e "${YELLOW}📊 Estimated monthly costs:${NC}"
    echo "• A10 (1 instance): ~$250/month"
    echo "• A100-40GB (1 instance): ~$900/month"
    echo "• A100-80GB (1 instance): ~$1,200/month"
}

# Parse command line arguments
COMMAND=""
INSTANCES=2
MEMORY="32gb"
GPU_TYPE="a100-40gb"

while [[ $# -gt 0 ]]; do
    case $1 in
        up|down|stop|start|status|auto|manual|cost)
            COMMAND="$1"
            shift
            ;;
        --instances)
            INSTANCES="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

# Check authentication
check_auth

# Execute command
case $COMMAND in
    up)
        scale_up $INSTANCES
        ;;
    down)
        scale_down
        ;;
    stop)
        stop_machines
        ;;
    start)
        start_machines
        ;;
    status)
        show_status
        ;;
    auto)
        enable_autoscaling
        ;;
    cost)
        show_cost_optimization
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac

echo -e "${GREEN}✅ Operation completed successfully${NC}"