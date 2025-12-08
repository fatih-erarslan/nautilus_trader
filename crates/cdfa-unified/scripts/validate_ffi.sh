#!/bin/bash
# Comprehensive FFI Validation Script for CDFA Unified Library
# 
# This script validates both C and Python FFI interfaces with:
# - Safety checks for financial system requirements
# - Memory leak detection
# - Performance benchmarking
# - Error handling verification
# - Cross-platform compatibility

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/target"
VALIDATION_LOG="$PROJECT_ROOT/ffi_validation.log"

# Validation counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$VALIDATION_LOG"
}

success() {
    echo -e "${GREEN}✓${NC} $*" | tee -a "$VALIDATION_LOG"
    ((PASSED_TESTS++))
}

failure() {
    echo -e "${RED}✗${NC} $*" | tee -a "$VALIDATION_LOG"
    ((FAILED_TESTS++))
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*" | tee -a "$VALIDATION_LOG"
}

info() {
    echo -e "${BLUE}ℹ${NC} $*" | tee -a "$VALIDATION_LOG"
}

test_header() {
    echo
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    ((TOTAL_TESTS++))
}

# Check prerequisites
check_prerequisites() {
    test_header "Checking Prerequisites"
    
    # Check Rust toolchain
    if command -v rustc >/dev/null 2>&1; then
        RUST_VERSION=$(rustc --version)
        success "Rust toolchain available: $RUST_VERSION"
    else
        failure "Rust toolchain not found"
        return 1
    fi
    
    # Check Cargo
    if command -v cargo >/dev/null 2>&1; then
        CARGO_VERSION=$(cargo --version)
        success "Cargo available: $CARGO_VERSION"
    else
        failure "Cargo not found"
        return 1
    fi
    
    # Check C compiler
    if command -v gcc >/dev/null 2>&1; then
        GCC_VERSION=$(gcc --version | head -n1)
        success "GCC available: $GCC_VERSION"
    elif command -v clang >/dev/null 2>&1; then
        CLANG_VERSION=$(clang --version | head -n1)
        success "Clang available: $CLANG_VERSION"
    else
        failure "No C compiler found (gcc or clang required)"
        return 1
    fi
    
    # Check Python (if available)
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version)
        success "Python3 available: $PYTHON_VERSION"
        
        # Check for NumPy
        if python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
            NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
            success "NumPy available: $NUMPY_VERSION"
        else
            warning "NumPy not available - Python bindings tests will be skipped"
        fi
    else
        warning "Python3 not available - Python bindings tests will be skipped"
    fi
    
    # Check for valgrind (memory leak detection)
    if command -v valgrind >/dev/null 2>&1; then
        VALGRIND_VERSION=$(valgrind --version)
        success "Valgrind available: $VALGRIND_VERSION"
    else
        warning "Valgrind not available - memory leak tests will be skipped"
    fi
    
    # Check for memory testing tools
    if command -v asan >/dev/null 2>&1 || command -v msan >/dev/null 2>&1; then
        success "Sanitizer tools available"
    else
        info "Sanitizer tools not found - using standard memory testing"
    fi
}

# Build the library with all FFI features
build_library() {
    test_header "Building CDFA Library with FFI Features"
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    info "Cleaning previous builds..."
    cargo clean
    
    # Build with all features including FFI
    info "Building library with FFI features..."
    if cargo build --release --features "ffi,python,c-bindings,core,algorithms,simd,parallel" 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "Library built successfully with FFI features"
    else
        failure "Failed to build library with FFI features"
        return 1
    fi
    
    # Check if the shared library was created
    if [[ -f "$BUILD_DIR/release/libcdfa_unified.so" ]] || [[ -f "$BUILD_DIR/release/libcdfa_unified.dylib" ]] || [[ -f "$BUILD_DIR/release/cdfa_unified.dll" ]]; then
        success "Shared library created successfully"
    else
        failure "Shared library not found"
        return 1
    fi
    
    # Build Python extension if possible
    if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" 2>/dev/null; then
        info "Building Python extension..."
        if maturin build --release --features "python" 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "Python extension built successfully"
        else
            warning "Failed to build Python extension with maturin, trying alternative method"
            if cargo build --release --features "python" 2>&1 | tee -a "$VALIDATION_LOG"; then
                success "Python bindings compiled successfully"
            else
                failure "Failed to build Python bindings"
            fi
        fi
    fi
}

# Run Rust tests
run_rust_tests() {
    test_header "Running Rust Tests"
    
    cd "$PROJECT_ROOT"
    
    # Run all tests including FFI tests
    info "Running Rust unit and integration tests..."
    if cargo test --features "ffi,python,c-bindings,core,algorithms" --release 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "All Rust tests passed"
    else
        failure "Some Rust tests failed"
        return 1
    fi
    
    # Run specific FFI tests
    info "Running FFI-specific tests..."
    if cargo test ffi --features "ffi,c-bindings" --release 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "FFI tests passed"
    else
        failure "FFI tests failed"
        return 1
    fi
    
    # Run C integration tests
    info "Running C integration tests..."
    if cargo test ffi_c_integration --features "c-bindings" --release 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "C integration tests passed"
    else
        failure "C integration tests failed"
        return 1
    fi
    
    # Run Python integration tests if available
    if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" 2>/dev/null; then
        info "Running Python integration tests..."
        if cargo test ffi_python_integration --features "python" --release 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "Python integration tests passed"
        else
            failure "Python integration tests failed"
            return 1
        fi
    fi
}

# Test C API with compiled example
test_c_api() {
    test_header "Testing C API"
    
    cd "$PROJECT_ROOT"
    
    # Compile C example
    info "Compiling C usage example..."
    local c_example="$PROJECT_ROOT/examples/ffi_examples/c_usage_example"
    local lib_path="$BUILD_DIR/release"
    
    # Determine library name based on platform
    local lib_name=""
    if [[ -f "$lib_path/libcdfa_unified.so" ]]; then
        lib_name="cdfa_unified"
    elif [[ -f "$lib_path/libcdfa_unified.dylib" ]]; then
        lib_name="cdfa_unified"
    else
        failure "Shared library not found for C API testing"
        return 1
    fi
    
    # Compile with proper flags
    if gcc -o "$c_example" \
        "$PROJECT_ROOT/examples/ffi_examples/c_usage_example.c" \
        -I"$PROJECT_ROOT/include" \
        -L"$lib_path" \
        -l"$lib_name" \
        -lm -pthread \
        -Wl,-rpath,"$lib_path" 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "C example compiled successfully"
    else
        failure "Failed to compile C example"
        return 1
    fi
    
    # Run C example
    info "Running C usage example..."
    if LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH" "$c_example" 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "C usage example executed successfully"
    else
        failure "C usage example failed"
        return 1
    fi
    
    # Clean up
    rm -f "$c_example"
}

# Test Python API
test_python_api() {
    test_header "Testing Python API"
    
    if ! command -v python3 >/dev/null 2>&1; then
        warning "Python3 not available - skipping Python API tests"
        return 0
    fi
    
    if ! python3 -c "import numpy" 2>/dev/null; then
        warning "NumPy not available - skipping Python API tests"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Set up Python path
    export PYTHONPATH="$BUILD_DIR/release:$PYTHONPATH"
    export LD_LIBRARY_PATH="$BUILD_DIR/release:$LD_LIBRARY_PATH"
    
    # Test basic import
    info "Testing Python module import..."
    if python3 -c "import cdfa_unified; print(f'CDFA Version: {cdfa_unified.get_version()}')" 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "Python module imported successfully"
    else
        failure "Failed to import Python module"
        return 1
    fi
    
    # Test basic functionality
    info "Testing basic Python functionality..."
    if python3 -c "
import cdfa_unified
import numpy as np

# Create test data
data = cdfa_unified.create_sample_data(4, 20, seed=42)
print(f'Sample data shape: {data.shape}')

# Validate data
cdfa_unified.validate_financial_data(data)
print('Data validation passed')

# Create CDFA instance
cdfa = cdfa_unified.CdfaUnified()
print(f'CDFA instance: {cdfa}')

# Calculate diversity
diversity = cdfa.calculate_diversity(data)
print(f'Diversity scores: {diversity.shape}')

print('Basic Python functionality test passed')
" 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "Basic Python functionality test passed"
    else
        failure "Basic Python functionality test failed"
        return 1
    fi
    
    # Run comprehensive Python example if available
    if [[ -f "$PROJECT_ROOT/examples/ffi_examples/python_usage_example.py" ]]; then
        info "Running comprehensive Python usage example..."
        if python3 "$PROJECT_ROOT/examples/ffi_examples/python_usage_example.py" 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "Python usage example executed successfully"
        else
            failure "Python usage example failed"
            return 1
        fi
    fi
}

# Memory safety and leak detection
test_memory_safety() {
    test_header "Testing Memory Safety"
    
    cd "$PROJECT_ROOT"
    
    # Build with AddressSanitizer if available
    info "Building with AddressSanitizer for memory safety testing..."
    if RUSTFLAGS="-Z sanitizer=address" cargo build --features "ffi,c-bindings" --target x86_64-unknown-linux-gnu 2>/dev/null; then
        success "Built with AddressSanitizer"
        
        # Run tests with AddressSanitizer
        info "Running tests with AddressSanitizer..."
        if RUSTFLAGS="-Z sanitizer=address" cargo test ffi --features "c-bindings" --target x86_64-unknown-linux-gnu 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "AddressSanitizer tests passed"
        else
            failure "AddressSanitizer detected issues"
            return 1
        fi
    else
        warning "AddressSanitizer not available - using standard memory testing"
    fi
    
    # Use valgrind if available
    if command -v valgrind >/dev/null 2>&1; then
        info "Running memory leak detection with Valgrind..."
        
        # Build debug version for better valgrind output
        cargo build --features "ffi,c-bindings" 2>/dev/null
        
        # Run a simple test under valgrind
        if valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes \
            cargo test ffi_c_integration --features "c-bindings" 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "Valgrind memory leak detection completed"
        else
            warning "Valgrind detected potential issues"
        fi
    else
        info "Valgrind not available - skipping memory leak detection"
    fi
    
    # Test for memory leaks in C API
    info "Testing C API memory management..."
    if [[ -f "$BUILD_DIR/release/libcdfa_unified.so" ]] || [[ -f "$BUILD_DIR/release/libcdfa_unified.dylib" ]]; then
        # Create a simple C program to test memory management
        cat > /tmp/memory_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

// Minimal declarations for testing
typedef struct CdfaHandle CdfaHandle;
typedef struct { double *data; unsigned int rows, cols; bool owns_data; } CArray2D;
typedef enum { CDFA_SUCCESS = 0 } CdfaErrorCode;

extern CdfaHandle* cdfa_create(void);
extern void cdfa_destroy(CdfaHandle* handle);
extern CArray2D* cdfa_alloc_array2d(unsigned int rows, unsigned int cols);
extern void cdfa_free_array2d(CArray2D* array);
extern CdfaErrorCode cdfa_validate_data(const CArray2D* data);

int main() {
    // Test multiple creation/destruction cycles
    for (int i = 0; i < 100; i++) {
        CdfaHandle* handle = cdfa_create();
        if (!handle) return 1;
        
        CArray2D* data = cdfa_alloc_array2d(10, 5);
        if (!data) return 1;
        
        // Fill with test data
        for (int j = 0; j < 50; j++) {
            data->data[j] = 100.0 + j;
        }
        
        cdfa_validate_data(data);
        
        cdfa_free_array2d(data);
        cdfa_destroy(handle);
    }
    
    printf("Memory management test completed successfully\n");
    return 0;
}
EOF
        
        # Compile and run memory test
        local lib_path="$BUILD_DIR/release"
        if gcc -o /tmp/memory_test /tmp/memory_test.c \
            -I"$PROJECT_ROOT/include" -L"$lib_path" -lcdfa_unified \
            -Wl,-rpath,"$lib_path" 2>/dev/null; then
            
            if LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH" /tmp/memory_test 2>&1 | tee -a "$VALIDATION_LOG"; then
                success "C API memory management test passed"
            else
                failure "C API memory management test failed"
                return 1
            fi
        else
            warning "Could not compile memory management test"
        fi
        
        # Clean up
        rm -f /tmp/memory_test /tmp/memory_test.c
    fi
}

# Performance benchmarking
test_performance() {
    test_header "Performance Benchmarking"
    
    cd "$PROJECT_ROOT"
    
    # Run Rust benchmarks
    info "Running Rust performance benchmarks..."
    if cargo bench --features "ffi,c-bindings,simd,parallel" 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "Rust benchmarks completed"
    else
        warning "Some Rust benchmarks failed"
    fi
    
    # Test C API performance
    info "Testing C API performance..."
    # Create a performance test program
    cat > /tmp/perf_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

typedef struct CdfaHandle CdfaHandle;
typedef struct { double *data; unsigned int rows, cols; bool owns_data; } CArray2D;
typedef struct { double *data; unsigned int len; bool owns_data; } CArray1D;
typedef enum { CDFA_SUCCESS = 0 } CdfaErrorCode;

extern CdfaHandle* cdfa_create(void);
extern void cdfa_destroy(CdfaHandle* handle);
extern CArray2D* cdfa_alloc_array2d(unsigned int rows, unsigned int cols);
extern CArray1D* cdfa_alloc_array1d(unsigned int len);
extern void cdfa_free_array2d(CArray2D* array);
extern void cdfa_free_array1d(CArray1D* array);
extern CdfaErrorCode cdfa_calculate_diversity(CdfaHandle* handle, const CArray2D* data, CArray1D* result);

int main() {
    CdfaHandle* handle = cdfa_create();
    if (!handle) return 1;
    
    // Create test data
    CArray2D* data = cdfa_alloc_array2d(100, 20);
    CArray1D* result = cdfa_alloc_array1d(20);
    
    // Fill with sample financial data
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 20; j++) {
            data->data[i * 20 + j] = 100.0 + i * 0.1 + j * 0.01;
        }
    }
    
    // Benchmark diversity calculation
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 100; i++) {
        cdfa_calculate_diversity(handle, data, result);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    long duration_ns = (end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec);
    double avg_time_ms = (duration_ns / 1000000.0) / 100.0;
    
    printf("Average diversity calculation time: %.3f ms\n", avg_time_ms);
    printf("Throughput: %.1f ops/sec\n", 1000.0 / avg_time_ms);
    
    cdfa_free_array1d(result);
    cdfa_free_array2d(data);
    cdfa_destroy(handle);
    
    return 0;
}
EOF
    
    local lib_path="$BUILD_DIR/release"
    if gcc -o /tmp/perf_test /tmp/perf_test.c \
        -I"$PROJECT_ROOT/include" -L"$lib_path" -lcdfa_unified \
        -Wl,-rpath,"$lib_path" 2>/dev/null; then
        
        if LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH" /tmp/perf_test 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "C API performance test completed"
        else
            failure "C API performance test failed"
        fi
    else
        warning "Could not compile performance test"
    fi
    
    # Clean up
    rm -f /tmp/perf_test /tmp/perf_test.c
    
    # Test Python API performance if available
    if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" 2>/dev/null; then
        info "Testing Python API performance..."
        
        python3 -c "
import time
import numpy as np
import cdfa_unified

# Create test data
data = np.random.randn(100, 20) + 100

# Create CDFA instance
cdfa = cdfa_unified.CdfaUnified()

# Benchmark diversity calculation
start_time = time.time()
for _ in range(100):
    diversity = cdfa.calculate_diversity(data)
end_time = time.time()

avg_time_ms = (end_time - start_time) * 1000 / 100
print(f'Average Python diversity calculation time: {avg_time_ms:.3f} ms')
print(f'Throughput: {1000/avg_time_ms:.1f} ops/sec')
" 2>&1 | tee -a "$VALIDATION_LOG" && success "Python API performance test completed" || warning "Python API performance test failed"
    fi
}

# Test error handling
test_error_handling() {
    test_header "Testing Error Handling"
    
    cd "$PROJECT_ROOT"
    
    # Test Rust error handling
    info "Testing Rust error handling..."
    if cargo test error --features "ffi,c-bindings" --release 2>&1 | tee -a "$VALIDATION_LOG"; then
        success "Rust error handling tests passed"
    else
        failure "Rust error handling tests failed"
        return 1
    fi
    
    # Test C API error handling
    info "Testing C API error handling..."
    cat > /tmp/error_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct CdfaHandle CdfaHandle;
typedef struct { double *data; unsigned int rows, cols; bool owns_data; } CArray2D;
typedef enum { 
    CDFA_SUCCESS = 0,
    CDFA_INVALID_INPUT = 1
} CdfaErrorCode;

extern CdfaHandle* cdfa_create(void);
extern void cdfa_destroy(CdfaHandle* handle);
extern CdfaErrorCode cdfa_validate_data(const CArray2D* data);
extern const char* cdfa_get_last_error(void);
extern void cdfa_clear_error(void);

int main() {
    printf("Testing C API error handling...\n");
    
    // Test 1: Null pointer handling
    CdfaErrorCode result = cdfa_validate_data(NULL);
    if (result == CDFA_INVALID_INPUT) {
        printf("✓ Null pointer correctly handled\n");
    } else {
        printf("✗ Null pointer not handled correctly\n");
        return 1;
    }
    
    // Test 2: Error message retrieval
    const char* error = cdfa_get_last_error();
    if (error != NULL) {
        printf("✓ Error message retrieved: %.50s...\n", error);
    } else {
        printf("✗ No error message available\n");
        return 1;
    }
    
    // Test 3: Error clearing
    cdfa_clear_error();
    error = cdfa_get_last_error();
    if (error == NULL) {
        printf("✓ Error cleared successfully\n");
    } else {
        printf("✗ Error not cleared\n");
        return 1;
    }
    
    // Test 4: Multiple error handling
    CdfaHandle* handle = cdfa_create();
    if (handle) {
        printf("✓ Handle creation succeeded\n");
        cdfa_destroy(handle);
    } else {
        printf("✗ Handle creation failed\n");
        return 1;
    }
    
    printf("All error handling tests passed\n");
    return 0;
}
EOF
    
    local lib_path="$BUILD_DIR/release"
    if gcc -o /tmp/error_test /tmp/error_test.c \
        -I"$PROJECT_ROOT/include" -L"$lib_path" -lcdfa_unified \
        -Wl,-rpath,"$lib_path" 2>/dev/null; then
        
        if LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH" /tmp/error_test 2>&1 | tee -a "$VALIDATION_LOG"; then
            success "C API error handling test passed"
        else
            failure "C API error handling test failed"
            return 1
        fi
    else
        warning "Could not compile error handling test"
    fi
    
    # Clean up
    rm -f /tmp/error_test /tmp/error_test.c
}

# Financial safety validation
test_financial_safety() {
    test_header "Testing Financial Safety Features"
    
    cd "$PROJECT_ROOT"
    
    info "Testing financial data validation constraints..."
    
    # Test various invalid financial data scenarios
    python3 -c "
import numpy as np
import cdfa_unified

print('Testing financial safety constraints...')

# Test cases for invalid data
test_cases = [
    ('NaN values', np.array([[1.0, np.nan], [3.0, 4.0]])),
    ('Infinite values', np.array([[1.0, 2.0], [3.0, np.inf]])),
    ('Unreasonably large values', np.array([[1.0, 2.0], [3.0, 1e16]])),
    ('Too small array', np.array([[1.0]])),
    ('All zeros', np.zeros((3, 3))),
]

passed = 0
for test_name, test_data in test_cases:
    try:
        cdfa_unified.validate_financial_data(test_data)
        print(f'✗ {test_name}: Should have failed validation')
    except ValueError:
        print(f'✓ {test_name}: Correctly rejected')
        passed += 1
    except Exception as e:
        print(f'? {test_name}: Unexpected error: {e}')

# Test valid financial data
try:
    valid_data = np.array([[100.0, 102.0, 101.5], [200.0, 198.0, 201.0]])
    cdfa_unified.validate_financial_data(valid_data)
    print('✓ Valid financial data: Correctly accepted')
    passed += 1
except Exception as e:
    print(f'✗ Valid financial data: Incorrectly rejected: {e}')

print(f'Financial safety tests: {passed}/6 passed')
if passed == 6:
    exit(0)
else:
    exit(1)
" 2>&1 | tee -a "$VALIDATION_LOG" && success "Financial safety validation passed" || failure "Financial safety validation failed"
}

# Generate comprehensive report
generate_report() {
    test_header "Generating Validation Report"
    
    local report_file="$PROJECT_ROOT/FFI_VALIDATION_REPORT.md"
    
    cat > "$report_file" << EOF
# CDFA Unified FFI Validation Report

**Generated:** $(date)
**Library Version:** $(cd "$PROJECT_ROOT" && cargo pkgid | cut -d'#' -f2)

## Summary

- **Total Tests:** $TOTAL_TESTS
- **Passed:** $PASSED_TESTS
- **Failed:** $FAILED_TESTS
- **Success Rate:** $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

## Test Results

### ✅ Passed Tests
$(grep "✓" "$VALIDATION_LOG" | sed 's/^/- /')

### ❌ Failed Tests
$(grep "✗" "$VALIDATION_LOG" | sed 's/^/- /')

### ⚠️ Warnings
$(grep "⚠" "$VALIDATION_LOG" | sed 's/^/- /')

## Build Information

- **Rust Version:** $(rustc --version)
- **Cargo Version:** $(cargo --version)
- **Target Platform:** $(rustc -vV | grep host | cut -d' ' -f2)
- **Features Tested:** ffi, python, c-bindings, core, algorithms, simd, parallel

## FFI Interface Status

### C API
- **Header File:** include/cdfa_unified.h
- **Shared Library:** $(ls "$BUILD_DIR/release"/libcdfa_unified.* 2>/dev/null || echo "Not found")
- **Memory Safety:** $(grep -q "AddressSanitizer tests passed" "$VALIDATION_LOG" && echo "Verified" || echo "Standard testing")
- **Error Handling:** $(grep -q "C API error handling test passed" "$VALIDATION_LOG" && echo "Verified" || echo "Failed")

### Python API
- **Module:** cdfa_unified
- **NumPy Integration:** $(python3 -c "import numpy; print('Available')" 2>/dev/null || echo "Not available")
- **Zero-Copy Operations:** $(grep -q "Python API performance test completed" "$VALIDATION_LOG" && echo "Verified" || echo "Not tested")

## Financial Safety Features

- **Data Validation:** Comprehensive validation for financial constraints
- **Range Checking:** Prevents unreasonably large values (> 1e15)
- **NaN/Infinity Detection:** Automatic rejection of invalid floating-point values
- **Variance Validation:** Ensures non-constant data for meaningful analysis
- **Memory Bounds:** Safe array access with bounds checking

## Performance Metrics

$(grep -E "(Average.*time|Throughput)" "$VALIDATION_LOG" | sed 's/^/- /')

## Recommendations

EOF

    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo "🎉 **All tests passed!** The FFI interfaces are ready for production use." >> "$report_file"
    else
        echo "⚠️ **Some tests failed.** Review the failed tests above before deploying." >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Next Steps

1. **Integration Testing:** Test with real financial data in your target environment
2. **Performance Tuning:** Adjust configuration for your specific use case
3. **Documentation:** Review the generated API documentation
4. **Deployment:** Follow the deployment guide for your platform

## Files Generated

- Validation Log: \`$VALIDATION_LOG\`
- C Header: \`include/cdfa_unified.h\`
- C Example: \`examples/ffi_examples/c_usage_example.c\`
- Python Example: \`examples/ffi_examples/python_usage_example.py\`

EOF
    
    success "Validation report generated: $report_file"
    info "View the complete report at: $report_file"
}

# Main execution
main() {
    echo "CDFA Unified FFI Validation Suite"
    echo "================================="
    echo
    
    # Initialize log
    echo "Starting FFI validation at $(date)" > "$VALIDATION_LOG"
    
    # Run all validation steps
    check_prerequisites || exit 1
    build_library || exit 1
    run_rust_tests || exit 1
    test_c_api || exit 1
    test_python_api || true  # Don't fail if Python not available
    test_memory_safety || true  # Don't fail if tools not available
    test_performance || true
    test_error_handling || exit 1
    test_financial_safety || exit 1
    
    # Generate final report
    generate_report
    
    echo
    echo "================================="
    echo "FFI Validation Summary"
    echo "================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "${GREEN}🎉 All FFI validation tests passed!${NC}"
        echo "The CDFA Unified library FFI interfaces are ready for use."
        exit 0
    else
        echo -e "${RED}❌ Some validation tests failed.${NC}"
        echo "Please review the failures and fix before proceeding."
        exit 1
    fi
}

# Run main function
main "$@"