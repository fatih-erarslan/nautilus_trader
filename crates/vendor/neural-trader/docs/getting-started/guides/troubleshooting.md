# Troubleshooting Guide

Comprehensive troubleshooting guide for the AI News Trading Platform with Neural Forecasting integration.

## Quick Diagnostics

### System Health Check

Run this comprehensive health check script to identify common issues:

```bash
#!/bin/bash
# quick_diagnostics.sh

echo "=== AI News Trading Platform Diagnostics ==="
echo "Date: $(date)"
echo "System: $(uname -a)"
echo

# Check Python environment
echo "=== Python Environment ==="
python --version
which python
echo "Virtual environment: $VIRTUAL_ENV"
echo

# Check GPU availability
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
    echo "CUDA Version: $(nvcc --version 2>/dev/null | grep "release" | cut -d' ' -f6)"
else
    echo "❌ NVIDIA GPU not detected"
fi
echo

# Check Python packages
echo "=== Critical Packages ==="
python -c "
packages = ['torch', 'neuralforecast', 'fastmcp', 'numpy', 'pandas']
for pkg in packages:
    try:
        __import__(pkg)
        exec(f'import {pkg}; print(f\"✓ {pkg}: {getattr({pkg}, \"__version__\", \"unknown\")}\")') 
    except ImportError:
        print(f'❌ {pkg}: not installed')
"
echo

# Check services
echo "=== Service Status ==="
curl -s http://localhost:3000/health &>/dev/null && echo "✓ MCP Server: Running" || echo "❌ MCP Server: Not responding"
curl -s http://localhost:8080/health &>/dev/null && echo "✓ Claude-Flow: Running" || echo "❌ Claude-Flow: Not responding"
echo

# Check databases
echo "=== Database Connectivity ==="
pg_isready -h localhost -p 5432 &>/dev/null && echo "✓ PostgreSQL: Connected" || echo "❌ PostgreSQL: Not connected"
redis-cli ping &>/dev/null && echo "✓ Redis: Connected" || echo "❌ Redis: Not connected"
echo

# Check disk space
echo "=== System Resources ==="
df -h | grep -E "(Filesystem|/dev/)"
echo
free -h
echo

echo "=== Network Connectivity ==="
ping -c 1 8.8.8.8 &>/dev/null && echo "✓ Internet: Connected" || echo "❌ Internet: No connection"
```

Save as `quick_diagnostics.sh`, make executable with `chmod +x quick_diagnostics.sh`, and run to get system overview.

## Common Issues and Solutions

### 1. Installation Issues

#### Python Package Conflicts

**Problem**: Conflicting package versions or dependencies

**Symptoms**:
```
ERROR: pip has conflicting dependencies
ImportError: cannot import name 'xyz' from 'package'
```

**Solutions**:

```bash
# Option 1: Clean installation
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements-mcp.txt

# Option 2: Force reinstall
pip install --force-reinstall --no-cache-dir -r requirements-mcp.txt

# Option 3: Use conda for better dependency resolution
conda create -n ai-news-trader python=3.10
conda activate ai-news-trader
pip install -r requirements-mcp.txt
```

#### CUDA/GPU Issues

**Problem**: CUDA not detected or version mismatch

**Symptoms**:
```python
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
```

**Solutions**:

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU detection
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

#### Permission Issues

**Problem**: Permission denied errors

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:

```bash
# Fix ownership
sudo chown -R $USER:$USER /workspaces/ai-news-trader

# Make scripts executable
chmod +x claude-flow
chmod +x *.py

# Fix file permissions
find . -type f -name "*.py" -exec chmod 644 {} \;
find . -type f -name "*.sh" -exec chmod 755 {} \;
```

### 2. Runtime Issues

#### MCP Server Not Starting

**Problem**: MCP server fails to start or crashes immediately

**Symptoms**:
```
Connection refused on port 3000
MCP server exited with code 1
```

**Diagnostic Steps**:

```bash
# Check if port is already in use
lsof -i :3000

# Start server with debug logging
python mcp_server_enhanced.py --debug --log-level DEBUG

# Check server logs
tail -f logs/mcp_server.log

# Test server manually
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}'
```

**Common Solutions**:

```bash
# Kill processes using port 3000
sudo kill -9 $(lsof -t -i:3000)

# Start server on different port
export MCP_SERVER_PORT=3001
python mcp_server_enhanced.py

# Check for missing dependencies
pip install fastmcp pydantic

# Reset configuration
rm -rf .mcp_cache/
python mcp_server_enhanced.py --reset-config
```

#### Neural Forecasting Errors

**Problem**: Neural forecasting models fail to load or predict

**Symptoms**:
```
ModuleNotFoundError: No module named 'neuralforecast'
RuntimeError: Expected all tensors to be on the same device
CUDA out of memory
```

**Solutions**:

```bash
# Install neural forecasting with GPU support
pip install neuralforecast[gpu]

# Check neuralforecast installation
python -c "
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
print('NeuralForecast imported successfully')
"

# Test basic neural forecasting
python -c "
import torch
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# Create test data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'ds': dates,
    'unique_id': 'TEST',
    'y': np.random.randn(100).cumsum()
})

# Test model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NHITS(
    input_size=30, 
    h=7, 
    max_epochs=5,
    accelerator=device
)
nf = NeuralForecast(models=[model], freq='D')

print('Testing neural forecasting...')
nf.fit(data)
forecasts = nf.predict()
print(f'✓ Test successful. Forecast shape: {forecasts.shape}')
"
```

#### GPU Memory Issues

**Problem**: GPU runs out of memory during training or inference

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions**:

```python
# Reduce batch size
export NEURAL_FORECAST_BATCH_SIZE=16

# Clear GPU cache
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"

# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Set memory fraction
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.6)
    print('Memory fraction set to 60%')
"
```

#### Claude-Flow Issues

**Problem**: Claude-Flow commands fail or hang

**Symptoms**:
```
./claude-flow: command not found
Command timed out after 5 minutes
```

**Solutions**:

```bash
# Fix claude-flow executable
chmod +x claude-flow

# Check if claude-flow is in PATH
which claude-flow || echo "Claude-flow not in PATH"

# Test basic claude-flow functionality
./claude-flow --help

# Reset claude-flow configuration
rm -rf .claude/
./claude-flow init --neural-forecast

# Check claude-flow status
./claude-flow status --verbose
```

### 3. Performance Issues

#### Slow Neural Forecasting

**Problem**: Neural forecasting takes too long

**Symptoms**:
- Forecasts taking >30 seconds
- High CPU usage, low GPU usage
- System becomes unresponsive

**Diagnostic Steps**:

```bash
# Check GPU utilization
nvidia-smi dmon -s u -d 1

# Profile neural forecasting
python -m cProfile -o forecast_profile.stats -c "
from src.forecasting.neural_forecast_integration import NeuralForecastEngine
engine = NeuralForecastEngine(gpu_acceleration=True)
# ... run forecast test
"

# Analyze profile
python -c "
import pstats
p = pstats.Stats('forecast_profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

**Optimization Solutions**:

```python
# Optimize model configuration
config = {
    "model": {
        "batch_size": 64,  # Increase batch size
        "input_size": 56,  # Reduce input size
        "max_epochs": 50   # Reduce epochs for faster training
    },
    "gpu": {
        "mixed_precision": True,  # Enable mixed precision
        "memory_fraction": 0.8
    }
}

# Use model caching
export NEURAL_FORECAST_CACHE=true

# Optimize data loading
export NEURAL_FORECAST_WORKERS=4
```

#### High Memory Usage

**Problem**: System uses excessive memory

**Symptoms**:
```
MemoryError: Unable to allocate array
System becomes slow or unresponsive
```

**Solutions**:

```bash
# Monitor memory usage
htop
free -h

# Check for memory leaks
ps aux --sort=-%mem | head -10

# Reduce model complexity
export NEURAL_FORECAST_BATCH_SIZE=16
export NEURAL_FORECAST_INPUT_SIZE=56

# Enable garbage collection
python -c "
import gc
gc.set_debug(gc.DEBUG_STATS)
gc.collect()
"
```

#### Network Connectivity Issues

**Problem**: API calls fail or timeout

**Symptoms**:
```
ConnectionError: HTTPSConnectionPool
TimeoutError: timed out
```

**Solutions**:

```bash
# Test network connectivity
ping api.example.com
curl -I https://api.example.com

# Check firewall settings
sudo ufw status
sudo iptables -L

# Increase timeout settings
export MCP_TIMEOUT=60
export REQUEST_TIMEOUT=30

# Use proxy if needed
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080
```

### 4. Configuration Issues

#### Environment Variables Not Loading

**Problem**: Environment variables not being read

**Symptoms**:
```
Configuration value 'None' is invalid
KeyError: 'NEURAL_FORECAST_GPU'
```

**Solutions**:

```bash
# Check if .env file exists and is readable
ls -la .env*
cat .env

# Source environment variables manually
source .env
export $(cat .env | xargs)

# Verify environment variables
env | grep NEURAL
env | grep MCP

# Create missing .env file
cat > .env << EOF
NEURAL_FORECAST_GPU=true
NEURAL_FORECAST_DEVICE=cuda
MCP_SERVER_PORT=3000
MCP_NEURAL_ENABLED=true
EOF
```

#### Configuration File Issues

**Problem**: Configuration files missing or invalid

**Symptoms**:
```
FileNotFoundError: 'config/neural_forecast.yaml'
yaml.scanner.ScannerError: invalid syntax
```

**Solutions**:

```bash
# Create missing directories
mkdir -p config/

# Validate YAML syntax
python -c "
import yaml
with open('config/neural_forecast.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('✓ YAML syntax is valid')
"

# Create default configuration
cat > config/neural_forecast.yaml << EOF
models:
  nhits:
    input_size: 168
    horizon: 30
    max_epochs: 100
    batch_size: 32

gpu:
  enabled: true
  memory_fraction: 0.8
EOF
```

### 5. Database Issues

#### PostgreSQL Connection Issues

**Problem**: Cannot connect to PostgreSQL database

**Symptoms**:
```
psycopg2.OperationalError: could not connect to server
FATAL: password authentication failed
```

**Solutions**:

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
pg_isready -h localhost -p 5432

# Connect with specific user
psql -h localhost -U ai_trader -d ai_news_trader

# Reset password
sudo -u postgres psql -c "ALTER USER ai_trader PASSWORD 'new_password';"

# Check pg_hba.conf authentication
sudo cat /etc/postgresql/14/main/pg_hba.conf | grep -v "^#"
```

#### Redis Connection Issues

**Problem**: Cannot connect to Redis

**Symptoms**:
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions**:

```bash
# Check Redis status
sudo systemctl status redis-server

# Test Redis connection
redis-cli ping

# Check Redis configuration
redis-cli CONFIG GET "*"

# Restart Redis
sudo systemctl restart redis-server
```

### 6. Docker Issues

#### Container Won't Start

**Problem**: Docker containers fail to start

**Symptoms**:
```
docker: Error response from daemon
Container exited with code 125
```

**Solutions**:

```bash
# Check Docker status
sudo systemctl status docker

# Build image with verbose output
docker build --no-cache -t ai-news-trader:debug .

# Run container with debug
docker run -it --rm ai-news-trader:debug /bin/bash

# Check container logs
docker logs container_name

# Fix permissions in container
docker run --user $(id -u):$(id -g) ai-news-trader:latest
```

#### GPU Not Available in Container

**Problem**: GPU not accessible inside Docker container

**Symptoms**:
```
nvidia-smi: command not found
CUDA not available in container
```

**Solutions**:

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Run container with GPU support
docker run --gpus all ai-news-trader:latest

# Check GPU in container
docker run --gpus all ai-news-trader:latest nvidia-smi

# Use GPU-enabled base image
FROM nvidia/cuda:11.8-devel-ubuntu20.04
```

## Error Code Reference

### MCP Server Error Codes

| Code | Name | Description | Solution |
|------|------|-------------|----------|
| -32700 | Parse Error | Invalid JSON | Check JSON syntax |
| -32600 | Invalid Request | Invalid request object | Verify request format |
| -32601 | Method Not Found | Method does not exist | Check method name |
| -32602 | Invalid Params | Invalid parameters | Validate parameter types |
| -32603 | Internal Error | Server internal error | Check server logs |
| -32001 | Model Not Found | Neural model not found | Reload models or check paths |
| -32002 | Strategy Error | Trading strategy error | Check strategy configuration |
| -32003 | Data Error | Market data error | Verify data sources |
| -32004 | Auth Error | Authentication failed | Check API keys |

### Neural Forecasting Error Codes

| Error Type | Common Causes | Solutions |
|------------|---------------|-----------|
| `ModelNotTrainedError` | Model not fitted | Call `fit()` before `predict()` |
| `GPUNotAvailableError` | CUDA not installed | Install CUDA or use CPU |
| `InvalidDataError` | Wrong data format | Check DataFrame structure |
| `MemoryError` | Insufficient memory | Reduce batch size or model size |
| `TimeoutError` | Training timeout | Reduce epochs or increase timeout |

### System Exit Codes

| Code | Description | Action |
|------|-------------|--------|
| 0 | Success | Normal execution |
| 1 | General error | Check logs for details |
| 2 | Configuration error | Fix configuration files |
| 3 | Neural model error | Check model files |
| 4 | GPU error | Check CUDA installation |
| 5 | MCP server error | Check MCP configuration |
| 6 | Memory error | Reduce memory usage |
| 7 | Network error | Check connectivity |

## Debugging Tools and Techniques

### Logging Configuration

#### Enable Debug Logging

```python
# debug_logging.py
import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Enable specific loggers
logging.getLogger('neuralforecast').setLevel(logging.DEBUG)
logging.getLogger('fastmcp').setLevel(logging.DEBUG)
logging.getLogger('src.forecasting').setLevel(logging.DEBUG)

# Test logging
logger = logging.getLogger(__name__)
logger.debug("Debug logging enabled")
```

### Performance Profiling

#### Profile Neural Forecasting

```python
# profile_neural.py
import cProfile
import pstats
import io
from src.forecasting.neural_forecast_integration import NeuralForecastEngine

def profile_forecast():
    engine = NeuralForecastEngine(gpu_acceleration=True)
    # Add test data and forecasting code here
    pass

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    
    profile_forecast()
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    print(s.getvalue())
```

#### Memory Profiling

```python
# memory_profile.py
from memory_profiler import profile
import psutil
import os

@profile
def forecast_with_memory_tracking():
    # Your forecasting code here
    pass

def monitor_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    monitor_memory()
    forecast_with_memory_tracking()
    monitor_memory()
```

### Network Debugging

#### Test API Endpoints

```python
# test_endpoints.py
import requests
import json
import time

def test_mcp_endpoint(method, params=None):
    url = "http://localhost:3000/mcp"
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": 1
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()
        
        print(f"Method: {method}")
        print(f"Status: {response.status_code}")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Response: {response.json()}")
        print("-" * 50)
        
    except requests.exceptions.RequestException as e:
        print(f"Error testing {method}: {e}")

# Test various endpoints
endpoints = [
    "ping",
    ("quick_analysis", {"symbol": "AAPL"}),
    ("list_strategies", {}),
]

for endpoint in endpoints:
    if isinstance(endpoint, tuple):
        test_mcp_endpoint(endpoint[0], endpoint[1])
    else:
        test_mcp_endpoint(endpoint)
```

## Advanced Troubleshooting

### System Resource Monitoring

#### Real-time Resource Monitor

```bash
#!/bin/bash
# monitor_resources.sh

echo "Starting resource monitoring..."
echo "Press Ctrl+C to stop"

while true; do
    clear
    echo "=== System Resources - $(date) ==="
    echo
    
    # CPU usage
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'
    
    # Memory usage
    echo
    echo "Memory Usage:"
    free -h | grep -E "(Mem|Swap)"
    
    # Disk usage
    echo
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)"
    
    # GPU usage (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo
        echo "GPU Usage:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    fi
    
    # Network connections
    echo
    echo "Network Connections:"
    netstat -an | grep -E "(3000|8080)" | wc -l
    
    sleep 5
done
```

### Database Debugging

#### PostgreSQL Query Analysis

```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check active connections
SELECT pid, usename, application_name, client_addr, state, query_start, query
FROM pg_stat_activity
WHERE state = 'active';

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Log Analysis Scripts

#### Automated Log Analysis

```python
# analyze_logs.py
import re
import json
from collections import defaultdict, Counter
from datetime import datetime

def analyze_log_file(filename):
    """Analyze log file for patterns and issues"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Pattern matching
    error_pattern = re.compile(r'ERROR|CRITICAL|Exception|Traceback')
    warning_pattern = re.compile(r'WARNING|WARN')
    performance_pattern = re.compile(r'took (\d+\.?\d*)ms|duration: (\d+\.?\d*)s')
    
    errors = []
    warnings = []
    performance_metrics = []
    
    for i, line in enumerate(lines):
        if error_pattern.search(line):
            # Get context (previous and next lines)
            context_start = max(0, i-2)
            context_end = min(len(lines), i+3)
            context = ''.join(lines[context_start:context_end])
            errors.append(context)
        
        if warning_pattern.search(line):
            warnings.append(line.strip())
        
        perf_match = performance_pattern.search(line)
        if perf_match:
            duration = float(perf_match.group(1) or perf_match.group(2))
            performance_metrics.append(duration)
    
    # Generate report
    report = {
        'total_lines': len(lines),
        'errors': len(errors),
        'warnings': len(warnings),
        'avg_performance': sum(performance_metrics) / len(performance_metrics) if performance_metrics else 0,
        'error_samples': errors[:5],  # First 5 errors
        'warning_samples': warnings[:5]  # First 5 warnings
    }
    
    return report

# Usage
if __name__ == "__main__":
    report = analyze_log_file('logs/ai-news-trader.log')
    print(json.dumps(report, indent=2))
```

### Emergency Recovery Procedures

#### Complete System Reset

```bash
#!/bin/bash
# emergency_reset.sh

echo "⚠️  EMERGENCY SYSTEM RESET ⚠️"
echo "This will reset the entire system to default state"
read -p "Are you sure? (type 'RESET' to confirm): " confirm

if [ "$confirm" != "RESET" ]; then
    echo "Reset cancelled"
    exit 1
fi

echo "Starting emergency reset..."

# Stop all services
echo "Stopping services..."
sudo systemctl stop ai-news-trader
sudo systemctl stop claude-flow
sudo systemctl stop nginx

# Clear caches
echo "Clearing caches..."
rm -rf .mcp_cache/
rm -rf .claude/cache/
redis-cli FLUSHALL

# Reset database (if needed)
read -p "Reset database? (y/N): " reset_db
if [ "$reset_db" = "y" ]; then
    echo "Resetting database..."
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS ai_news_trader_prod;"
    sudo -u postgres psql -c "CREATE DATABASE ai_news_trader_prod OWNER ai_trader;"
fi

# Recreate virtual environment
echo "Recreating Python environment..."
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements-mcp.txt

# Reset configuration
echo "Resetting configuration..."
cp config/neural_forecast.yaml.default config/neural_forecast.yaml
cp .env.example .env

# Restart services
echo "Restarting services..."
sudo systemctl start ai-news-trader
sudo systemctl start claude-flow
sudo systemctl start nginx

# Wait and test
sleep 10
echo "Testing system..."
./quick_diagnostics.sh

echo "Emergency reset complete!"
```

#### Backup Current State

```bash
#!/bin/bash
# backup_current_state.sh

BACKUP_DIR="/opt/backups/emergency_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "Backing up current state to $BACKUP_DIR"

# Backup configuration
cp -r config/ $BACKUP_DIR/
cp .env $BACKUP_DIR/
cp -r .claude/ $BACKUP_DIR/

# Backup database
pg_dump -h localhost -U ai_trader ai_news_trader_prod | gzip > $BACKUP_DIR/database.sql.gz

# Backup models
cp -r models/ $BACKUP_DIR/

# Backup logs
cp -r logs/ $BACKUP_DIR/

# Create backup manifest
cat > $BACKUP_DIR/manifest.txt << EOF
Backup created: $(date)
System: $(uname -a)
Python: $(python --version)
Git commit: $(git rev-parse HEAD)
Files backed up:
$(find $BACKUP_DIR -type f | wc -l) files
$(du -sh $BACKUP_DIR)
EOF

echo "Backup complete: $BACKUP_DIR"
echo "Manifest:"
cat $BACKUP_DIR/manifest.txt
```

## Getting Help

### Information to Collect

When reporting issues, please collect this information:

```bash
#!/bin/bash
# collect_debug_info.sh

DEBUG_FILE="debug_info_$(date +%Y%m%d_%H%M%S).txt"

echo "Collecting debug information..."
echo "Output file: $DEBUG_FILE"

{
    echo "=== SYSTEM INFORMATION ==="
    date
    uname -a
    lsb_release -a 2>/dev/null || echo "lsb_release not available"
    
    echo
    echo "=== PYTHON ENVIRONMENT ==="
    python --version
    which python
    echo "Virtual environment: $VIRTUAL_ENV"
    pip list
    
    echo
    echo "=== GPU INFORMATION ==="
    nvidia-smi 2>/dev/null || echo "NVIDIA GPU not available"
    nvcc --version 2>/dev/null || echo "NVCC not available"
    
    echo
    echo "=== DISK SPACE ==="
    df -h
    
    echo
    echo "=== MEMORY USAGE ==="
    free -h
    
    echo
    echo "=== NETWORK CONNECTIVITY ==="
    curl -I http://localhost:3000/health 2>/dev/null || echo "MCP server not responding"
    curl -I http://localhost:8080/health 2>/dev/null || echo "Claude-flow not responding"
    
    echo
    echo "=== RECENT LOGS ==="
    echo "--- MCP Server Logs ---"
    tail -50 logs/mcp_server.log 2>/dev/null || echo "No MCP server logs found"
    
    echo
    echo "--- System Logs ---"
    sudo journalctl -u ai-news-trader --since "1 hour ago" -n 50 2>/dev/null || echo "No system logs found"
    
    echo
    echo "=== CONFIGURATION ==="
    echo "--- Environment Variables ---"
    env | grep -E "(NEURAL|MCP|CLAUDE)" | sort
    
    echo
    echo "--- Config Files ---"
    if [ -f config/neural_forecast.yaml ]; then
        echo "Neural forecast config exists"
    else
        echo "Neural forecast config missing"
    fi
    
    if [ -f .env ]; then
        echo ".env file exists"
    else
        echo ".env file missing"
    fi
    
} > $DEBUG_FILE

echo "Debug information collected in: $DEBUG_FILE"
echo "Please include this file when reporting issues"
```

### Community Support

- **GitHub Issues**: [Create an issue](https://github.com/your-org/ai-news-trader/issues)
- **Discord**: Join our community server
- **Documentation**: Check the [docs](../README.md)
- **Stack Overflow**: Tag questions with `ai-news-trader` and `neural-forecasting`

### Professional Support

For enterprise customers:
- **Priority Support**: Email support@your-company.com
- **Emergency Hotline**: +1-xxx-xxx-xxxx
- **Dedicated Slack Channel**: Available for enterprise customers

---

This troubleshooting guide covers the most common issues you might encounter. If your issue isn't covered here, please collect debug information using the provided scripts and reach out to our support channels.