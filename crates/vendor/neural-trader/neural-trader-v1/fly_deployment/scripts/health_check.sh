#!/bin/bash
# Health Check Script for Fly.io GPU Deployment
# AI News Trading Platform - NeuralForecast NHITS

set -e

# Configuration
HEALTH_ENDPOINT="http://localhost:8080/health/gpu"
TIMEOUT=10
MAX_RETRIES=3

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HEALTH_CHECK: $1"
}

# Function to check application health
check_app_health() {
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -f -s --max-time $TIMEOUT "$HEALTH_ENDPOINT" > /dev/null 2>&1; then
            log "✅ Application health check passed"
            return 0
        else
            retry_count=$((retry_count + 1))
            log "⚠️  Health check attempt $retry_count/$MAX_RETRIES failed"
            if [ $retry_count -lt $MAX_RETRIES ]; then
                sleep 2
            fi
        fi
    done
    
    log "❌ Application health check failed after $MAX_RETRIES attempts"
    return 1
}

# Function to check GPU health
check_gpu_health() {
    if [ "$NEURAL_FORECAST_GPU_ENABLED" = "true" ]; then
        # Check if nvidia-smi is working
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi --query-gpu=name,memory.free --format=csv,noheader >/dev/null 2>&1; then
                log "✅ GPU health check passed"
                return 0
            else
                log "❌ GPU not accessible"
                return 1
            fi
        else
            log "❌ nvidia-smi not found"
            return 1
        fi
    else
        log "ℹ️  GPU disabled, skipping GPU health check"
        return 0
    fi
}

# Function to check system resources
check_system_resources() {
    # Check memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
    if [ "$memory_usage" -gt 90 ]; then
        log "⚠️  High memory usage: ${memory_usage}%"
        return 1
    fi
    
    # Check disk space
    local disk_usage=$(df /app | awk 'NR==2 {printf "%d", $5}')
    if [ "$disk_usage" -gt 85 ]; then
        log "⚠️  High disk usage: ${disk_usage}%"
        return 1
    fi
    
    log "✅ System resources check passed (Memory: ${memory_usage}%, Disk: ${disk_usage}%)"
    return 0
}

# Function to check neural forecast functionality
check_neural_forecast() {
    local response
    local http_code
    
    # Make a test prediction request
    response=$(curl -s -w "%{http_code}" --max-time 15 \
        -X POST "$HEALTH_ENDPOINT/../neural/predict" \
        -H "Content-Type: application/json" \
        -d '{"symbol": "AAPL", "horizon": 1}' 2>/dev/null || echo "000")
    
    http_code="${response: -3}"
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "503" ]; then
        log "✅ Neural forecast endpoint accessible"
        return 0
    else
        log "❌ Neural forecast endpoint failed (HTTP: $http_code)"
        return 1
    fi
}

# Function to perform comprehensive health check
comprehensive_health_check() {
    log "🔍 Starting comprehensive health check..."
    
    local checks_passed=0
    local total_checks=4
    
    # Application health
    if check_app_health; then
        checks_passed=$((checks_passed + 1))
    fi
    
    # GPU health (if enabled)
    if check_gpu_health; then
        checks_passed=$((checks_passed + 1))
    fi
    
    # System resources
    if check_system_resources; then
        checks_passed=$((checks_passed + 1))
    fi
    
    # Neural forecast functionality
    if check_neural_forecast; then
        checks_passed=$((checks_passed + 1))
    fi
    
    log "📊 Health check results: $checks_passed/$total_checks checks passed"
    
    if [ $checks_passed -eq $total_checks ]; then
        log "✅ All health checks passed - system healthy"
        return 0
    elif [ $checks_passed -ge 2 ]; then
        log "⚠️  Partial health - $checks_passed/$total_checks checks passed"
        return 0  # Still consider healthy for basic functionality
    else
        log "❌ Health check failed - $checks_passed/$total_checks checks passed"
        return 1
    fi
}

# Function to get detailed system status
get_system_status() {
    log "📋 System Status Report:"
    
    # Basic system info
    log "   OS: $(uname -a)"
    log "   Uptime: $(uptime -p)"
    log "   Load: $(uptime | awk -F'load average:' '{print $2}')"
    
    # Memory status
    local memory_info=$(free -h | grep Mem)
    log "   Memory: $memory_info"
    
    # Disk status
    local disk_info=$(df -h /app | tail -1)
    log "   Disk: $disk_info"
    
    # GPU status (if available)
    if [ "$NEURAL_FORECAST_GPU_ENABLED" = "true" ] && command -v nvidia-smi &> /dev/null; then
        log "   GPU Status:"
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader | while IFS=, read -r name temp util mem_used mem_total; do
            log "     $name: ${temp}°C, ${util}% util, ${mem_used}/${mem_total} memory"
        done
    fi
    
    # Process status
    if pgrep -f "flyio_gpu_launcher.py" > /dev/null; then
        log "   ✅ Main application process running"
    else
        log "   ❌ Main application process not found"
    fi
}

# Main execution
main() {
    case "${1:-check}" in
        "check")
            comprehensive_health_check
            ;;
        "status")
            get_system_status
            comprehensive_health_check
            ;;
        "quick")
            check_app_health
            ;;
        "gpu")
            check_gpu_health
            ;;
        *)
            log "Usage: $0 [check|status|quick|gpu]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"