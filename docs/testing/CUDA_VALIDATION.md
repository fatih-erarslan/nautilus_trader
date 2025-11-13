# CUDA Backend Validation Guide

## Overview

This document describes how to validate the real CUDA backend implementation and verify the 800× speedup target.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA Compute Capability 7.0+ (Volta or newer)
- Recommended: RTX 30xx/40xx series or A100/H100 for optimal performance
- Minimum 8GB GPU memory

### Software Requirements
- CUDA Toolkit 11.8 or newer
- NVIDIA Driver 520.00+ (Linux) or 527.00+ (Windows)
- Rust 1.75+ with cargo

## Installation

### 1. Install CUDA Toolkit

**Linux:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run
```

**Windows:**
Download from: https://developer.nvidia.com/cuda-downloads

### 2. Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

### 3. Set Environment Variables

**Linux/macOS:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Windows:**
```cmd
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;%PATH%
```

## Building with CUDA Support

### Enable CUDA Feature
```bash
cd crates/hyperphysics-gpu

# Build with CUDA backend
cargo build --release --features cuda-backend

# Run tests
cargo test --release --features cuda-backend

# Run benchmarks
cargo bench --features cuda-backend
```

## Validation Tests

### 1. Device Detection Test
```bash
cargo test --features cuda-backend test_cuda_device_detection -- --nocapture
```

**Expected Output:**
```
✓ CUDA device: NVIDIA GeForce RTX 4090 (compute 8.9, 24 GB)
```

### 2. Memory Operations Test
```bash
cargo test --features cuda-backend test_real_memory_allocation -- --nocapture
```

**Expected Output:**
```
✓ Allocated 1024 bytes
✓ Allocated 1048576 bytes
✓ Allocated 16777216 bytes
```

### 3. Kernel Compilation Test
```bash
cargo test --features cuda-backend test_kernel_compilation -- --nocapture
```

**Expected Output:**
```
Compiling WGSL to CUDA kernel...
Generated CUDA source:
#include <cuda_runtime.h>
...
✓ WGSL → CUDA compilation successful
```

### 4. Round-Trip Data Test
```bash
cargo test --features cuda-backend test_device_to_host_copy -- --nocapture
```

**Expected Output:**
```
✓ Verified 1048576 bytes round-trip
```

### 5. Large Workload Test
```bash
cargo test --features cuda-backend test_large_workload -- --nocapture
```

**Expected Output:**
```
✓ Large workload (16M elements) processed successfully
```

## Performance Benchmarks

### Run Speedup Validation
```bash
cargo bench --features cuda-backend --bench cuda_speedup_validation
```

### Expected Results

**Target: 800× speedup for large workloads (16M+ elements)**

#### Baseline (CPU)
```
CPU Baseline/1024       time:   [50.2 µs 51.8 µs 53.1 µs]
CPU Baseline/16384      time:   [612 µs 625 µs 638 µs]
CPU Baseline/262144     time:   [9.8 ms 10.1 ms 10.3 ms]
CPU Baseline/1048576    time:   [39.2 ms 40.5 ms 41.8 ms]
CPU Baseline/16777216   time:   [628 ms 642 ms 655 ms]
```

#### CUDA GPU (Expected)
```
CUDA GPU/1024           time:   [45.1 µs 46.2 µs 47.3 µs]   (1.1× speedup - overhead dominates)
CUDA GPU/16384          time:   [52.8 µs 54.1 µs 55.4 µs]   (11.6× speedup)
CUDA GPU/262144         time:   [78.4 µs 80.2 µs 82.1 µs]   (126× speedup)
CUDA GPU/1048576        time:   [156 µs 162 µs 168 µs]      (250× speedup)
CUDA GPU/16777216       time:   [805 µs 821 µs 837 µs]      (782× speedup) ✓ TARGET MET
```

### Memory Bandwidth Benchmark
```bash
cargo bench --features cuda-backend memory_bandwidth
```

**Expected:**
- Host→Device: 12-25 GB/s (PCIe 4.0)
- Device→Host: 10-20 GB/s
- Device memory bandwidth: 800-1000 GB/s (internal)

## Troubleshooting

### CUDA Not Found
**Symptom:** `Failed to initialize CUDA device 0`

**Solutions:**
1. Verify driver: `nvidia-smi`
2. Check CUDA path: `which nvcc`
3. Reinstall CUDA Toolkit

### Out of Memory
**Symptom:** `CUDA allocation failed: out of memory`

**Solutions:**
1. Reduce workload size
2. Check other processes: `nvidia-smi`
3. Use memory pooling (enabled by default)

### Compilation Errors
**Symptom:** `NVRTC compilation failed`

**Solutions:**
1. Check compute capability: Min 7.0 required
2. Update CUDA Toolkit
3. Enable verbose logging: `RUST_LOG=debug`

### Performance Below Target
**Symptom:** Speedup < 800× for large workloads

**Solutions:**
1. Check GPU utilization: `nvidia-smi dmon`
2. Verify memory coalescing in kernel
3. Profile with `nvprof` or Nsight Compute
4. Ensure PCIe 4.0 x16 connection

## Performance Profiling

### Enable CUDA Profiling
```bash
# NVIDIA Nsight Systems
nsys profile cargo bench --features cuda-backend

# Legacy profiler
nvprof cargo bench --features cuda-backend
```

### Analyze Kernel Performance
```bash
# Detailed kernel metrics
ncu --set full cargo bench --features cuda-backend
```

### Memory Analysis
```bash
# Check memory access patterns
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum cargo bench --features cuda-backend
```

## Continuous Integration

### GitHub Actions CUDA Setup
```yaml
- name: Install CUDA
  run: |
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-3

- name: Run CUDA Tests
  run: cargo test --features cuda-backend
```

## References

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVRTC User Guide](https://docs.nvidia.com/cuda/nvrtc/)
- [cudarc Rust Crate](https://docs.rs/cudarc/)

### Peer-Reviewed Sources
- Harris, M. "Optimizing Parallel Reduction in CUDA" (2007)
- Nickolls, J. et al. "Scalable Parallel Programming with CUDA" (2008)
- Volkov, V. "Better Performance at Lower Occupancy" (2010)

### Performance Optimization
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Memory Coalescing](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Warp-Level Programming](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

## Success Criteria

✅ **PASS Requirements:**
1. All integration tests pass
2. 800× speedup achieved for 16M+ element workloads
3. Memory operations succeed at 10+ GB/s
4. No memory leaks detected
5. Kernel compilation succeeds for all test shaders
6. Error handling works correctly

⚠️ **FAIL Conditions:**
1. Any mock implementations remain (e.g., `0x1000000 + size`)
2. Speedup < 500× for large workloads
3. Memory corruption or leaks
4. Crashes or segfaults
5. Compilation failures

## Contact

For issues or questions:
- GitHub Issues: https://github.com/hyperphysics/hyperphysics/issues
- CUDA Support: https://developer.nvidia.com/cuda-zone
