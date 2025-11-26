#!/bin/bash
# GPU Initialization Script for Fly.io Deployment
# AI News Trading Platform - NeuralForecast NHITS

set -e

echo "🚀 Initializing GPU environment for AI News Trading Platform..."

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check GPU availability
check_gpu() {
    log "Checking GPU availability..."
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        log "❌ nvidia-smi not found. GPU may not be available."
        return 1
    fi
    
    # Check GPU status
    if nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits; then
        log "✅ GPU detected and accessible"
        return 0
    else
        log "❌ GPU not accessible"
        return 1
    fi
}

# Function to optimize GPU settings
optimize_gpu() {
    log "Optimizing GPU settings..."
    
    # Set GPU performance mode
    if command -v nvidia-smi &> /dev/null; then
        # Set persistence mode for better performance
        sudo nvidia-smi -pm 1 2>/dev/null || log "⚠️  Could not set persistence mode (may require root)"
        
        # Set power management to maximum performance
        sudo nvidia-smi -pl 400 2>/dev/null || log "⚠️  Could not set power limit"
        
        # Set GPU clocks for optimal performance
        sudo nvidia-smi -ac 1215,1410 2>/dev/null || log "⚠️  Could not set application clocks"
    fi
    
    log "✅ GPU optimization completed"
}

# Function to verify CUDA environment
verify_cuda() {
    log "Verifying CUDA environment..."
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log "✅ CUDA version: $CUDA_VERSION"
    else
        log "⚠️  CUDA compiler not found"
    fi
    
    # Verify CUDA libraries
    if [ -d "$CUDA_HOME/lib64" ]; then
        log "✅ CUDA libraries found in $CUDA_HOME/lib64"
    else
        log "⚠️  CUDA libraries not found"
    fi
    
    # Check cuDNN
    if find /usr/local/cuda -name "libcudnn*" 2>/dev/null | head -1; then
        log "✅ cuDNN libraries found"
    else
        log "⚠️  cuDNN libraries not found"
    fi
}

# Function to test PyTorch GPU
test_pytorch_gpu() {
    log "Testing PyTorch GPU functionality..."
    
    python3 -c "
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {gpu.name}')
        print(f'  Memory: {gpu.total_memory / 1024**3:.1f} GB')
        print(f'  Compute capability: {gpu.major}.{gpu.minor}')
    
    # Test basic tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✅ GPU tensor operations working')
    
    # Test memory allocation
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    print(f'GPU memory allocated: {allocated:.1f} MB')
    
else:
    print('❌ CUDA not available')
    sys.exit(1)
" && log "✅ PyTorch GPU test passed" || log "❌ PyTorch GPU test failed"
}

# Function to initialize neural forecast models
init_neural_models() {
    log "Initializing NeuralForecast models..."
    
    python3 -c "
import os
import sys
sys.path.append('/app')

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    import torch
    
    print('✅ NeuralForecast imports successful')
    
    # Test model creation
    model = NHITS(
        h=24,
        input_size=168,
        max_steps=1,
        enable_progress_bar=False
    )
    print('✅ NHITS model creation successful')
    
    # Test GPU model transfer
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'✅ Using device: {device}')
    else:
        print('⚠️  Using CPU fallback')
        
except Exception as e:
    print(f'❌ NeuralForecast initialization failed: {e}')
    sys.exit(1)
" && log "✅ NeuralForecast initialization successful" || log "❌ NeuralForecast initialization failed"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up GPU monitoring..."
    
    # Create monitoring directories
    mkdir -p /app/logs/gpu
    mkdir -p /app/metrics
    
    # Start GPU monitoring in background
    python3 -c "
import subprocess
import os

# Start nvidia-smi monitoring
if os.path.exists('/usr/bin/nvidia-smi'):
    with open('/app/logs/gpu/nvidia_monitoring.log', 'w') as f:
        subprocess.Popen([
            'nvidia-smi', 
            '--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total',
            '--format=csv',
            '--loop=30'
        ], stdout=f, stderr=f)
    print('✅ GPU monitoring started')
else:
    print('⚠️  nvidia-smi not available for monitoring')
" &
    
    log "✅ Monitoring setup completed"
}

# Function to create GPU performance baseline
create_baseline() {
    log "Creating GPU performance baseline..."
    
    python3 -c "
import torch
import time
import json
import os

if not torch.cuda.is_available():
    print('❌ GPU not available for baseline')
    exit(1)

# Warm up GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
for _ in range(10):
    torch.matmul(x, y)
torch.cuda.synchronize()

# Benchmark matrix multiplication
sizes = [100, 500, 1000, 2000]
results = {}

for size in sizes:
    times = []
    for _ in range(5):
        x = torch.randn(size, size).cuda()
        y = torch.randn(size, size).cuda()
        
        torch.cuda.synchronize()
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    results[f'matmul_{size}x{size}'] = {
        'avg_time_ms': avg_time * 1000,
        'gflops': (2 * size**3) / (avg_time * 1e9)
    }

# Save baseline
os.makedirs('/app/metrics', exist_ok=True)
with open('/app/metrics/gpu_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)

print('✅ GPU baseline created')
print(json.dumps(results, indent=2))
" && log "✅ GPU baseline creation successful" || log "❌ GPU baseline creation failed"
}

# Main initialization sequence
main() {
    log "🚀 Starting GPU initialization sequence..."
    
    # Check if we're running with GPU support
    if [ "$NEURAL_FORECAST_GPU_ENABLED" != "true" ]; then
        log "⚠️  GPU support disabled, skipping GPU initialization"
        exec "$@"
        return
    fi
    
    # Run initialization steps
    check_gpu || {
        log "❌ GPU check failed, falling back to CPU mode"
        export NEURAL_FORECAST_GPU_ENABLED=false
        exec "$@"
        return
    }
    
    verify_cuda
    optimize_gpu
    test_pytorch_gpu
    init_neural_models
    setup_monitoring
    create_baseline
    
    log "✅ GPU initialization completed successfully"
    
    # Export GPU configuration for application
    export GPU_INITIALIZED=true
    export GPU_DEVICE_COUNT=$(nvidia-smi --list-gpus | wc -l)
    export GPU_MEMORY_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    
    log "🎯 GPU Environment Ready:"
    log "   - GPU Count: $GPU_DEVICE_COUNT"
    log "   - Total Memory: ${GPU_MEMORY_TOTAL}MB"
    log "   - CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
    
    # Continue with application startup
    if [ $# -gt 0 ] && [ "$1" != "&&" ]; then
        log "🚀 Starting application: $@"
        exec "$@"
    else
        log "✅ GPU initialization complete"
    fi
}

# Handle signals gracefully
trap 'log "Received signal, cleaning up..."; pkill -f nvidia-smi; exit 0' SIGTERM SIGINT

# Run main function
main "$@"