# CDFA Unified FFI Interfaces

This document provides comprehensive documentation for the Foreign Function Interface (FFI) implementations in the CDFA Unified library, enabling seamless integration with C/C++ applications and Python environments.

## Overview

The CDFA Unified library provides robust FFI interfaces designed specifically for financial systems with:

- **C API**: Complete C-compatible interface with proper memory management
- **Python Bindings**: PyO3-based bindings with NumPy integration for zero-copy operations
- **Financial Safety**: Comprehensive validation and error boundaries for financial data
- **Performance**: SIMD and parallel processing support through FFI
- **Memory Safety**: Rust's memory safety guarantees extended to FFI boundaries

## Quick Start

### C API

```c
#include "cdfa_unified.h"

// Create CDFA instance
CdfaHandle* handle = cdfa_create();

// Create and validate financial data
CArray2D* data = cdfa_alloc_array2d(4, 100);  // 4 assets, 100 time periods
// ... fill data with stock prices ...
cdfa_validate_data(data);

// Perform analysis
CAnalysisResult* result = NULL;
cdfa_analyze(handle, data, &result);

// Use results
printf("Execution time: %llu microseconds\n", result->execution_time_us);

// Clean up
cdfa_free_result(result);
cdfa_free_array2d(data);
cdfa_destroy(handle);
```

### Python API

```python
import cdfa_unified
import numpy as np

# Create CDFA instance
cdfa = cdfa_unified.CdfaUnified()

# Create financial data (zero-copy with NumPy)
data = np.random.randn(4, 100) + 100  # 4 assets, 100 periods

# Validate financial data
cdfa_unified.validate_financial_data(data)

# Perform analysis
result = cdfa.analyze(data)

# Access results
print(f"Execution time: {result.get_performance().execution_time_us} μs")
print(f"Patterns detected: {len(result.get_patterns())}")
```

## Features

### 🛡️ Financial System Safety

- **Data Validation**: Automatic validation of financial data constraints
- **Range Checking**: Prevention of unreasonably large values (> 1e15)
- **NaN/Infinity Detection**: Automatic rejection of invalid floating-point values
- **Variance Validation**: Ensures non-constant data for meaningful analysis
- **Memory Bounds**: Safe array access with comprehensive bounds checking

### 🚀 Performance Optimization

- **Zero-Copy Operations**: Direct memory access for NumPy arrays
- **SIMD Support**: Hardware acceleration for mathematical operations
- **Parallel Processing**: Multi-threaded analysis with configurable thread counts
- **Memory Efficiency**: Optimized memory allocation and caching strategies
- **Hardware Detection**: Automatic detection and utilization of available features

### 🔧 Robust Error Handling

- **Comprehensive Error Codes**: Detailed error classification for different failure modes
- **Thread-Safe Error Handling**: Thread-local error state management
- **Graceful Degradation**: Fallback strategies for unsupported operations
- **Memory Safety**: Protection against memory leaks and buffer overflows

## Building

### Prerequisites

- Rust 1.70+ with Cargo
- C compiler (GCC or Clang)
- Python 3.8+ with NumPy (for Python bindings)
- CMake (optional, for advanced builds)

### Build Commands

```bash
# Build with all FFI features
cargo build --release --features "ffi,python,c-bindings,core,algorithms,simd,parallel"

# Build C API only
cargo build --release --features "c-bindings,core,algorithms"

# Build Python bindings only
cargo build --release --features "python,core,algorithms"

# Build with maturin (Python packaging)
maturin build --release --features "python"
```

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `ffi` | Enables all FFI interfaces | `c-bindings`, `python` |
| `c-bindings` | C API interface | `libc` |
| `python` | Python bindings | `pyo3`, `numpy` |
| `core` | Core CDFA algorithms | - |
| `algorithms` | Advanced algorithms | `rustfft`, `ta` |
| `simd` | SIMD optimizations | `wide`, `pulp` |
| `parallel` | Parallel processing | `rayon`, `crossbeam` |

## API Reference

### C API

#### Core Types

```c
typedef struct CdfaHandle CdfaHandle;           // Opaque CDFA instance
typedef struct CArray2D CArray2D;               // 2D array for matrices
typedef struct CArray1D CArray1D;               // 1D array for vectors
typedef struct CCdfaConfig CCdfaConfig;         // Configuration structure
typedef struct CAnalysisResult CAnalysisResult; // Analysis results
```

#### Error Codes

```c
typedef enum {
    CDFA_SUCCESS = 0,                // Operation successful
    CDFA_INVALID_INPUT = 1,          // Invalid input parameters
    CDFA_DIMENSION_MISMATCH = 2,     // Array dimension mismatch
    CDFA_MATH_ERROR = 3,             // Mathematical computation error
    CDFA_NUMERICAL_ERROR = 4,        // Numerical instability
    // ... additional error codes
} CdfaErrorCode;
```

#### Core Functions

```c
// Lifecycle management
CdfaHandle* cdfa_create(void);
CdfaHandle* cdfa_create_with_config(const CCdfaConfig* config);
void cdfa_destroy(CdfaHandle* handle);

// Analysis operations
CdfaErrorCode cdfa_analyze(CdfaHandle* handle, const CArray2D* data, CAnalysisResult** result);
CdfaErrorCode cdfa_calculate_diversity(CdfaHandle* handle, const CArray2D* data, CArray1D* result);
CdfaErrorCode cdfa_apply_fusion(CdfaHandle* handle, const CArray1D* scores, const CArray2D* data, CArray1D* result);

// Validation and utilities
CdfaErrorCode cdfa_validate_data(const CArray2D* data);
const char* cdfa_get_version(void);
const char* cdfa_get_last_error(void);

// Memory management
CArray2D* cdfa_alloc_array2d(uint32_t rows, uint32_t cols);
CArray1D* cdfa_alloc_array1d(uint32_t len);
void cdfa_free_array2d(CArray2D* array);
void cdfa_free_array1d(CArray1D* array);
void cdfa_free_result(CAnalysisResult* result);
```

### Python API

#### Core Classes

```python
class CdfaUnified:
    """Main CDFA analysis interface"""
    def __init__(self, config: Optional[CdfaConfig] = None)
    def analyze(self, data: np.ndarray) -> AnalysisResult
    def calculate_diversity(self, data: np.ndarray) -> np.ndarray
    def apply_fusion(self, scores: np.ndarray, data: np.ndarray) -> np.ndarray
    
class CdfaConfig:
    """Configuration for CDFA operations"""
    num_threads: int
    enable_simd: bool
    enable_gpu: bool
    tolerance: float
    
class AnalysisResult:
    """Results from CDFA analysis"""
    def get_data(self) -> np.ndarray
    def get_metrics(self) -> Dict[str, float]
    def get_patterns(self) -> List[Pattern]
    def get_performance(self) -> PerformanceMetrics
```

#### Utility Functions

```python
def validate_financial_data(data: np.ndarray) -> None
def get_version() -> str
def create_sample_data(rows: int, cols: int, seed: Optional[int] = None) -> np.ndarray
```

## Examples

### Complete C Example

See [`examples/ffi_examples/c_usage_example.c`](examples/ffi_examples/c_usage_example.c) for a comprehensive C usage example including:

- Financial data creation and validation
- Error handling patterns
- Memory management best practices
- Performance measurement
- Multi-threaded usage

### Complete Python Example

See [`examples/ffi_examples/python_usage_example.py`](examples/ffi_examples/python_usage_example.py) for a comprehensive Python usage example including:

- NumPy integration patterns
- Time series data handling
- Batch processing
- Error handling
- Performance optimization
- Visualization with matplotlib

### Integration Examples

#### C++ Integration

```cpp
#include "cdfa_unified.h"
#include <memory>
#include <vector>

class CdfaWrapper {
    std::unique_ptr<CdfaHandle, decltype(&cdfa_destroy)> handle_;
    
public:
    CdfaWrapper() : handle_(cdfa_create(), cdfa_destroy) {
        if (!handle_) {
            throw std::runtime_error("Failed to create CDFA handle");
        }
    }
    
    std::vector<double> analyze(const std::vector<std::vector<double>>& data) {
        // Implementation with RAII and exception safety
        // ...
    }
};
```

#### NumPy Zero-Copy Integration

```python
import numpy as np
import cdfa_unified

# Create large financial dataset
prices = np.random.randn(1000, 100) + 100  # 1000 assets, 100 periods

# Zero-copy analysis (no data copying)
cdfa = cdfa_unified.CdfaUnified()
result = cdfa.analyze(prices)  # Direct memory access to NumPy array

# Access results without copying
analysis_data = result.get_data()  # Returns NumPy array view
```

## Testing and Validation

### Automated Testing

Run the comprehensive FFI validation suite:

```bash
./scripts/validate_ffi.sh
```

This script performs:

- ✅ Build verification with all FFI features
- ✅ Unit and integration test execution
- ✅ C API functionality testing
- ✅ Python bindings validation
- ✅ Memory safety analysis (AddressSanitizer, Valgrind)
- ✅ Performance benchmarking
- ✅ Error handling verification
- ✅ Financial safety constraint testing

### Manual Testing

```bash
# Test C API
gcc -o test_c examples/ffi_examples/c_usage_example.c \
    -I./include -L./target/release -lcdfa_unified
./test_c

# Test Python API
python3 examples/ffi_examples/python_usage_example.py

# Run specific test suites
cargo test ffi_c_integration --features c-bindings
cargo test ffi_python_integration --features python
```

### Memory Safety Testing

```bash
# AddressSanitizer
RUSTFLAGS="-Z sanitizer=address" cargo test --features ffi

# Valgrind
valgrind --tool=memcheck cargo test ffi_c_integration

# Memory leak detection
./scripts/validate_ffi.sh  # Includes comprehensive memory testing
```

## Performance Considerations

### Optimization Tips

1. **Use Zero-Copy Operations**: Leverage NumPy's memory layout for direct access
2. **Configure Thread Count**: Set `num_threads` based on your CPU cores
3. **Enable SIMD**: Ensure `enable_simd = true` for mathematical operations
4. **Batch Processing**: Process multiple datasets together for better cache utilization
5. **Memory Management**: Reuse arrays when possible to reduce allocation overhead

### Benchmarks

Typical performance on modern hardware:

| Operation | C API | Python API | Notes |
|-----------|-------|------------|-------|
| Diversity Calculation | ~2ms | ~3ms | 100×20 matrix |
| Full Analysis | ~15ms | ~18ms | Including pattern detection |
| Memory Allocation | ~0.1ms | ~0.05ms | 1000×100 matrix |

## Deployment

### C/C++ Applications

1. Include the header: `#include "cdfa_unified.h"`
2. Link the library: `-lcdfa_unified`
3. Set library path: `-L./target/release`
4. Runtime library path: `LD_LIBRARY_PATH=./target/release`

### Python Applications

```bash
# Install from wheel
pip install target/wheels/cdfa_unified-*.whl

# Or install in development mode
pip install -e .
```

### Docker Deployment

```dockerfile
FROM rust:1.75 as builder
COPY . /app
WORKDIR /app
RUN cargo build --release --features ffi

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3 python3-numpy
COPY --from=builder /app/target/release/libcdfa_unified.so /usr/lib/
COPY --from=builder /app/include/cdfa_unified.h /usr/include/
```

## Troubleshooting

### Common Issues

#### Build Errors

**Problem**: `error: could not find feature python`
**Solution**: Install PyO3 dependencies: `pip install maturin`

**Problem**: `undefined reference to cdfa_create`
**Solution**: Check library linking: `-L./target/release -lcdfa_unified`

#### Runtime Errors

**Problem**: `ImportError: No module named 'cdfa_unified'`
**Solution**: Set Python path: `export PYTHONPATH=./target/release:$PYTHONPATH`

**Problem**: `error while loading shared libraries`
**Solution**: Set library path: `export LD_LIBRARY_PATH=./target/release:$LD_LIBRARY_PATH`

#### Memory Issues

**Problem**: Segmentation faults in C code
**Solution**: Ensure proper memory management and check for null pointers

**Problem**: Memory leaks detected
**Solution**: Match every `cdfa_alloc_*` with corresponding `cdfa_free_*`

### Debug Mode

Build with debug symbols for troubleshooting:

```bash
cargo build --features ffi  # Debug build
RUST_BACKTRACE=1 ./your_program  # Full backtraces
RUST_LOG=debug ./your_program    # Detailed logging
```

## Contributing

### Adding New FFI Functions

1. Define the function in `src/ffi/mod.rs`
2. Add C declaration to `include/cdfa_unified.h`
3. Add Python binding in `src/ffi/python.rs`
4. Write tests in `tests/ffi_*_integration.rs`
5. Update documentation and examples

### Testing Guidelines

- All FFI functions must have corresponding tests
- Memory safety tests are required for C API functions
- Python bindings must include NumPy integration tests
- Financial data validation must be comprehensive

## Security Considerations

### Input Validation

- All external input is validated at FFI boundaries
- Financial data constraints are enforced
- Array bounds are checked for all operations
- Memory safety is guaranteed through Rust's type system

### Memory Safety

- No buffer overflows possible
- Automatic memory management prevents leaks
- Double-free protection in all deallocation functions
- Thread-safe error handling prevents race conditions

## License

This project is licensed under MIT OR Apache-2.0. See LICENSE files for details.

## Support

- **Documentation**: [Full API documentation](docs/)
- **Examples**: [Complete working examples](examples/ffi_examples/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

For more information, see the main [README.md](README.md) and [API documentation](docs/).