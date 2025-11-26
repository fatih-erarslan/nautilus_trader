#!/bin/bash

# GPU-Optimized Startup Script for Fly.io
# Initializes all GPU optimization components and starts the trading platform

set -euo pipefail

echo "🚀 Starting GPU-Optimized AI News Trading Platform on Fly.io"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check GPU availability
check_gpu() {
    log "🔍 Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        log "✅ GPU detected and accessible"
        return 0
    else
        log "⚠️  nvidia-smi not found, checking CUDA availability..."
        python -c "import cupy; print('GPU Available:', cupy.cuda.is_available()); print('GPU Count:', cupy.cuda.runtime.getDeviceCount())" || {
            log "❌ GPU not available, will run in CPU fallback mode"
            export ENABLE_CPU_FALLBACK=true
            export FORCE_CPU_MODE=true
            return 1
        }
    fi
}

# Function to initialize GPU memory
init_gpu_memory() {
    log "🧠 Initializing GPU memory management..."
    
    python -c "
import os
import cupy as cp
from gpu_acceleration.gpu_memory_manager import get_gpu_memory_manager

try:
    # Initialize memory manager
    manager = get_gpu_memory_manager()
    
    # Get memory info
    info = manager.get_memory_info()
    print(f'GPU Memory initialized: {info[\"total_allocated\"]:.2f}GB allocated, {info[\"total_free\"]:.2f}GB free')
    
    # Test allocation
    test_array = manager.allocate_array((1000, 1000), tag='startup_test')
    if test_array is not None:
        print('✅ GPU memory allocation test successful')
    else:
        print('⚠️  GPU memory allocation test failed')
        
except Exception as e:
    print(f'❌ GPU memory initialization failed: {str(e)}')
    exit(1)
"
}

# Function to initialize GPU monitoring
init_gpu_monitoring() {
    log "📊 Initializing GPU monitoring..."
    
    python -c "
from gpu_acceleration.gpu_monitor import get_gpu_monitor
from gpu_acceleration.flyio_optimizer import initialize_flyio_optimization

try:
    # Start GPU monitoring
    monitor = get_gpu_monitor()
    monitor.start_monitoring()
    print('✅ GPU monitoring started')
    
    # Initialize fly.io optimization
    workload_type = os.environ.get('WORKLOAD_TYPE', 'trading')
    result = initialize_flyio_optimization(workload_type)
    
    if result['status'] == 'success':
        print(f'✅ Fly.io optimization initialized for {workload_type} workload')
    else:
        print(f'⚠️  Fly.io optimization failed: {result.get(\"error\", \"Unknown error\")}')
        
except Exception as e:
    print(f'❌ GPU monitoring initialization failed: {str(e)}')
"
}

# Function to validate GPU optimization components
validate_gpu_components() {
    log "🔧 Validating GPU optimization components..."
    
    python -c "
from gpu_acceleration.flyio_gpu_config import initialize_flyio_gpu_config
from gpu_acceleration.mixed_precision import create_mixed_precision_manager
from gpu_acceleration.batch_optimizer import create_batch_processor
from gpu_acceleration.cpu_fallback import get_fallback_manager

try:
    # Test GPU configuration
    config_result = initialize_flyio_gpu_config()
    print(f'GPU Config: {config_result[\"status\"]}')
    
    # Test mixed precision
    mp_manager = create_mixed_precision_manager()
    print(f'Mixed Precision: {mp_manager.config.precision_mode.value} mode')
    
    # Test batch processor
    batch_processor = create_batch_processor('adaptive')
    print('Batch Processor: adaptive strategy initialized')
    
    # Test fallback manager
    fallback_manager = get_fallback_manager()
    status = fallback_manager.get_status()
    print(f'Fallback Manager: {status[\"current_mode\"]} mode')
    
    print('✅ All GPU components validated successfully')
    
except Exception as e:
    print(f'❌ GPU component validation failed: {str(e)}')
    exit(1)
"
}

# Function to set up health monitoring endpoints
setup_health_endpoints() {
    log "🏥 Setting up health monitoring endpoints..."
    
    # Start health monitoring server in background
    python -c "
import threading
import time
from flask import Flask, jsonify
from gpu_acceleration.gpu_monitor import get_gpu_monitor
from gpu_acceleration.gpu_memory_manager import get_gpu_memory_manager
from gpu_acceleration.cpu_fallback import get_fallback_manager

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/gpu-health')
def gpu_health():
    try:
        monitor = get_gpu_monitor()
        metrics = monitor.get_current_metrics()
        
        if metrics:
            return jsonify({
                'status': 'healthy',
                'gpu_utilization': metrics.gpu_utilization,
                'memory_utilization': metrics.memory_utilization,
                'temperature': metrics.temperature_c,
                'timestamp': metrics.timestamp
            })
        else:
            return jsonify({'status': 'no_metrics', 'timestamp': time.time()})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'timestamp': time.time()})

@app.route('/gpu-memory')
def gpu_memory():
    try:
        manager = get_gpu_memory_manager()
        info = manager.get_memory_info()
        return jsonify({
            'status': 'healthy',
            'memory_info': info,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'timestamp': time.time()})

@app.route('/performance-metrics')
def performance_metrics():
    try:
        monitor = get_gpu_monitor()
        summary = monitor.get_metrics_summary(5)  # Last 5 minutes
        fallback = get_fallback_manager()
        fallback_status = fallback.get_status()
        
        return jsonify({
            'status': 'healthy',
            'metrics_summary': summary,
            'fallback_status': fallback_status,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'timestamp': time.time()})

if __name__ == '__main__':
    # Run health server in background thread
    health_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8081, debug=False))
    health_thread.daemon = True
    health_thread.start()
    print('Health monitoring endpoints started on port 8081')
" &

    log "✅ Health monitoring endpoints started"
}

# Function to start the main application
start_application() {
    log "🎯 Starting main application with GPU optimization..."
    
    # Set final environment variables
    export PYTHONPATH="/app:$PYTHONPATH"
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_CACHE_DISABLE=0
    
    # Start the main application
    if [[ "${ENABLE_FLY_OPTIMIZATION:-true}" == "true" ]]; then
        log "🚁 Starting with Fly.io optimization enabled"
        python -m src.main --gpu-optimized --fly-mode --port 8080
    else
        log "🐍 Starting with basic GPU optimization"
        python -m src.main --gpu --port 8080
    fi
}

# Function to handle graceful shutdown
graceful_shutdown() {
    log "🛑 Received shutdown signal, gracefully stopping services..."
    
    # Stop GPU monitoring
    python -c "
from gpu_acceleration.gpu_monitor import get_gpu_monitor
from gpu_acceleration.flyio_optimizer import get_fly_optimizer

try:
    monitor = get_gpu_monitor()
    monitor.stop_monitoring()
    print('GPU monitoring stopped')
    
    optimizer = get_fly_optimizer()
    optimizer.stop_optimization()
    print('Fly.io optimization stopped')
    
except Exception as e:
    print(f'Error during shutdown: {str(e)}')
"
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    log "✅ Graceful shutdown completed"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap graceful_shutdown SIGTERM SIGINT

# Main startup sequence
main() {
    log "🔧 Starting GPU optimization initialization sequence..."
    
    # Check if we're running in Fly.io
    if [[ -n "${FLY_APP_NAME:-}" ]]; then
        log "🚁 Running on Fly.io: ${FLY_APP_NAME} in region ${FLY_REGION:-unknown}"
    else
        log "🐳 Running in local/container environment"
    fi
    
    # Step 1: Check GPU availability
    if check_gpu; then
        # Step 2: Initialize GPU memory
        init_gpu_memory
        
        # Step 3: Validate GPU components
        validate_gpu_components
        
        # Step 4: Initialize GPU monitoring
        init_gpu_monitoring
    else
        log "⚠️  GPU not available, running in CPU fallback mode"
        export FORCE_CPU_MODE=true
    fi
    
    # Step 5: Setup health endpoints
    setup_health_endpoints
    
    # Step 6: Wait a moment for all services to initialize
    log "⏳ Waiting for services to initialize..."
    sleep 5
    
    # Step 7: Start main application
    start_application
}

# Error handling
set +e
main "$@"
exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    log "❌ Application exited with code $exit_code"
    graceful_shutdown
fi

exit $exit_code