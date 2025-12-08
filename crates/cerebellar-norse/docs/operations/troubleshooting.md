# Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for common issues encountered with the Cerebellar-Norse neural network system.

## Quick Diagnostic Commands

### System Health Check
```bash
# Check service status
systemctl status cerebellar-norse

# Check application health endpoint
curl -s http://localhost:8080/health | jq

# Check resource usage
htop
nvidia-smi
df -h
free -h

# Check logs
journalctl -u cerebellar-norse -f
tail -f /var/log/cerebellar-norse/application.log
```

### Performance Diagnostics
```bash
# Check processing latency
curl -s http://localhost:8080/metrics | grep latency

# Monitor CUDA utilization
watch -n 1 nvidia-smi

# Check memory usage
cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable)"

# Network connectivity
netstat -tulpn | grep :8080
```

## Common Issues and Solutions

### 1. High Processing Latency

#### Symptoms
- Processing time > 10μs consistently
- API timeouts
- Degraded trading performance

#### Diagnosis
```bash
# Check CPU usage
top -p $(pgrep cerebellar-norse)

# Check GPU utilization
nvidia-smi dmon -s u

# Check memory allocation
pmap $(pgrep cerebellar-norse) | tail -1

# Check for memory leaks
valgrind --tool=massif ./target/release/cerebellar-norse
```

#### Root Causes and Solutions

**Cause: CUDA Memory Exhaustion**
```bash
# Check CUDA memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Solution: Reduce batch size in configuration
[performance]
batch_size = 512  # Reduce from 1024
cuda_streams = 2  # Reduce from 4
```

**Cause: CPU Thermal Throttling**
```bash
# Check CPU temperature
sensors | grep Core

# Solution: Improve cooling or reduce CPU frequency
echo 2000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
```

**Cause: Memory Fragmentation**
```bash
# Check memory fragmentation
cat /proc/buddyinfo

# Solution: Restart service or use memory compaction
echo 1 | sudo tee /proc/sys/vm/compact_memory
systemctl restart cerebellar-norse
```

### 2. Neural Network Convergence Issues

#### Symptoms
- Training loss not decreasing
- Erratic spike patterns
- Poor prediction accuracy

#### Diagnosis
```bash
# Check training metrics
curl -s http://localhost:8080/training/status/latest | jq

# Examine spike patterns
curl -s http://localhost:8080/neural/metrics | jq '.neural_metrics.spike_counts'

# Check weight distributions
cerebellar-norse --dump-weights | python analyze_weights.py
```

#### Solutions

**Learning Rate Too High**
```toml
[neural]
learning_rate = 0.0001  # Reduce from 0.01
```

**Network Size Mismatch**
```bash
# Check configuration consistency
cerebellar-norse --validate-config --config /etc/cerebellar-norse/production.toml
```

**Insufficient Training Data**
```bash
# Check training data size
wc -l /data/training/*.csv
# Minimum recommended: 10,000 samples per output class
```

### 3. CUDA Errors

#### Common CUDA Error Codes

**CUDA_ERROR_OUT_OF_MEMORY (2)**
```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Solutions:
# 1. Reduce neural network size
[neural]
granule_size = 2000000  # Reduce from 4000000

# 2. Enable memory pooling
[performance]
memory_pool_enabled = true
memory_pool_size = "20GB"

# 3. Use gradient checkpointing
[training]
gradient_checkpointing = true
```

**CUDA_ERROR_INVALID_DEVICE (101)**
```bash
# Check GPU availability
nvidia-smi -L

# Check CUDA driver
nvidia-smi

# Solution: Update GPU drivers
sudo apt update && sudo apt install nvidia-driver-525
```

**CUDA_ERROR_LAUNCH_FAILED (719)**
```bash
# Check kernel compilation
nvcc --version
dmesg | grep NVRM

# Solution: Recompile CUDA kernels
cargo clean
cargo build --release --features lightning-gpu
```

### 4. Memory Issues

#### Memory Leaks
```bash
# Monitor memory growth
while true; do
    ps -o pid,vsz,rss,comm -p $(pgrep cerebellar-norse)
    sleep 60
done

# Use memory profiler
valgrind --tool=memcheck --leak-check=full ./target/release/cerebellar-norse
```

#### Out of Memory (OOM)
```bash
# Check OOM killer logs
dmesg | grep -i "killed process"
journalctl | grep -i "out of memory"

# Check memory limits
ulimit -v
cat /proc/$(pgrep cerebellar-norse)/limits
```

**Solutions:**
```toml
[performance]
memory_limit = "28GB"  # Leave 4GB for system
garbage_collection_threshold = 0.8
```

### 5. Network and Connectivity Issues

#### API Timeouts
```bash
# Check connection limits
ss -tuln | grep :8080
cat /proc/sys/net/core/somaxconn

# Check request processing time
curl -w "@curl-format.txt" http://localhost:8080/health

# curl-format.txt content:
time_namelookup:  %{time_namelookup}\n
time_connect:     %{time_connect}\n
time_appconnect:  %{time_appconnect}\n
time_pretransfer: %{time_pretransfer}\n
time_redirect:    %{time_redirect}\n
time_starttransfer: %{time_starttransfer}\n
time_total:       %{time_total}\n
```

#### Solutions
```toml
[server]
worker_threads = 16
connection_timeout = 30
keep_alive_timeout = 75
max_connections = 10000
```

### 6. Configuration Issues

#### Invalid Configuration
```bash
# Validate configuration syntax
cerebellar-norse --config /etc/cerebellar-norse/production.toml --validate-only

# Check configuration schema
cerebellar-norse --dump-schema > config-schema.json
```

#### Common Configuration Errors

**Device Mismatch**
```toml
# Ensure GPU is available when using CUDA
[performance]
device = "cuda"  # But no GPU available

# Solution: Check GPU availability first
nvidia-smi || echo "No GPU available, using CPU"
```

**Resource Limits**
```toml
# Memory allocation exceeds available memory
[neural]
granule_size = 10000000  # Requires > 32GB RAM

# Solution: Calculate memory requirements
# granule_size * 4 bytes (f32) * connections_per_neuron < available_memory
```

## Performance Debugging

### Profiling Tools

#### CPU Profiling
```bash
# Install perf
sudo apt install linux-tools-common linux-tools-generic

# Profile CPU usage
sudo perf record -g ./target/release/cerebellar-norse
sudo perf report

# Profile specific function
sudo perf record -g -e cpu-cycles:u --call-graph dwarf ./target/release/cerebellar-norse
```

#### GPU Profiling
```bash
# Install NVIDIA Nsight Systems
wget https://developer.nvidia.com/nsight-systems
sudo dpkg -i nsight-systems-*.deb

# Profile GPU kernels
nsys profile --trace=cuda,openmp ./target/release/cerebellar-norse

# Analyze results
nsys-ui
```

#### Memory Profiling
```bash
# Install heaptrack
sudo apt install heaptrack

# Profile memory allocation
heaptrack ./target/release/cerebellar-norse
heaptrack_gui heaptrack.cerebellar-norse.*.gz
```

### Benchmark Validation

```bash
# Run performance benchmarks
cargo bench

# Compare with baseline
cargo bench -- --save-baseline baseline
# After changes:
cargo bench -- --baseline baseline

# Specific benchmark
cargo bench neuron_step_benchmark
```

## Log Analysis

### Log Levels and Patterns

#### Error Patterns
```bash
# CUDA errors
grep -E "CUDA_ERROR|cuMemAlloc|cuLaunchKernel" /var/log/cerebellar-norse/

# Memory errors
grep -E "OutOfMemory|allocation failed|segfault" /var/log/cerebellar-norse/

# Performance warnings
grep -E "latency.*exceeded|timeout|slow" /var/log/cerebellar-norse/
```

#### Structured Log Analysis
```bash
# Extract metrics from JSON logs
jq -r 'select(.level == "ERROR") | .message' /var/log/cerebellar-norse/app.log

# Performance metrics
jq -r 'select(.fields.processing_time_ns) | .fields.processing_time_ns' /var/log/cerebellar-norse/app.log
```

### Log Rotation and Management
```bash
# Check log rotation status
logrotate -d /etc/logrotate.d/cerebellar-norse

# Manual log rotation
logrotate -f /etc/logrotate.d/cerebellar-norse

# Log retention policy
[logging]
max_files = 30
max_size = "100MB"
compression = "gzip"
```

## Emergency Procedures

### Critical System Failure

1. **Immediate Actions**
   ```bash
   # Stop service to prevent data corruption
   sudo systemctl stop cerebellar-norse
   
   # Backup current state
   sudo cp -r /var/lib/cerebellar-norse /var/lib/cerebellar-norse.backup.$(date +%s)
   
   # Check system resources
   df -h
   free -h
   dmesg | tail -50
   ```

2. **Rapid Recovery**
   ```bash
   # Restore from known good configuration
   sudo cp /etc/cerebellar-norse/backup/production.toml /etc/cerebellar-norse/
   
   # Start with reduced capacity
   sudo sed -i 's/granule_size = 4000000/granule_size = 1000000/' /etc/cerebellar-norse/production.toml
   
   # Restart service
   sudo systemctl start cerebellar-norse
   ```

3. **Validation**
   ```bash
   # Verify service health
   curl -f http://localhost:8080/health
   
   # Check processing capability
   curl -X POST http://localhost:8080/neural/process \
     -H "Content-Type: application/json" \
     -d '{"price": 100.0, "volume": 1000.0, "timestamp": 1640995200000, "market_id": "TEST"}'
   ```

### Data Corruption

1. **Detection**
   ```bash
   # Check data integrity
   cerebellar-norse --verify-data /var/lib/cerebellar-norse/
   
   # Check model weights
   cerebellar-norse --validate-model /var/lib/cerebellar-norse/models/
   ```

2. **Recovery**
   ```bash
   # Restore from backup
   sudo systemctl stop cerebellar-norse
   sudo rm -rf /var/lib/cerebellar-norse/models/
   sudo cp -r /backup/cerebellar-norse/models/ /var/lib/cerebellar-norse/
   sudo chown -R cerebellar:cerebellar /var/lib/cerebellar-norse/
   sudo systemctl start cerebellar-norse
   ```

## Monitoring and Alerting

### Key Metrics to Monitor

```yaml
# Prometheus alerting rules
groups:
- name: cerebellar-norse
  rules:
  - alert: HighProcessingLatency
    expr: histogram_quantile(0.95, rate(processing_duration_seconds_bucket[5m])) > 0.00001
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Processing latency above 10μs"
      
  - alert: CUDAMemoryHigh
    expr: cuda_memory_used_bytes / cuda_memory_total_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "CUDA memory usage above 90%"
      
  - alert: NeuralConvergenceFailure
    expr: rate(training_loss[10m]) > 0
    for: 15m
    labels:
      severity: major
    annotations:
      summary: "Neural network not converging"
```

### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Cerebellar-Norse System Health",
    "panels": [
      {
        "title": "Processing Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(processing_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "CUDA Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cuda_memory_used_bytes",
            "legendFormat": "Used Memory"
          }
        ]
      }
    ]
  }
}
```

## Contact and Escalation

### Support Levels

1. **Level 1 (Operations Team)**
   - Service restarts
   - Configuration changes
   - Basic troubleshooting

2. **Level 2 (Engineering Team)**
   - Performance optimization
   - Neural network tuning
   - Complex diagnostics

3. **Level 3 (Architecture Team)**
   - System redesign
   - Critical bugs
   - Research and development

### Emergency Contacts

- **On-call Engineer**: +1-555-0123
- **Technical Lead**: +1-555-0124
- **DevOps Lead**: +1-555-0125

### Escalation Matrix

| Issue Severity | Response Time | Escalation |
|----------------|---------------|------------|
| Critical | 15 minutes | Immediate |
| Major | 1 hour | 2 hours |
| Minor | 4 hours | 24 hours |

---

*For additional support, consult the knowledge base or create a support ticket in the project management system.*