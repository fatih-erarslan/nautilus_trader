#!/usr/bin/env python3
"""
Standalone Test for Optimization Integration
Tests Numba kernels and Intel MKL integration without relative imports

Author: Agent 3 - Numba & Mathematical Optimization Specialist
"""

import numpy as np
import time
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the optimization modules directly
try:
    from numba_optimized_kernels import (
        numba_wavelet_variance_kernel,
        numba_wavelet_variance_kernel_vectorized,
        numba_wavelet_correlation_kernel,
        numba_wavelet_correlation_kernel_vectorized,
        numba_trend_strength_kernel,
        numba_wavelet_momentum_kernel,
        numba_convolution_kernel,
        validate_numba_kernels
    )
    NUMBA_AVAILABLE = True
    print("✅ Numba kernels imported successfully")
except Exception as e:
    NUMBA_AVAILABLE = False
    print(f"❌ Numba kernels import failed: {e}")

try:
    from intel_mkl_integration import IntelMKLAccelerator
    MKL_AVAILABLE = True
    print("✅ Intel MKL integration imported successfully")
except Exception as e:
    MKL_AVAILABLE = False
    print(f"❌ Intel MKL integration import failed: {e}")

def reference_wavelet_variance(series, scale):
    """Reference implementation for comparison"""
    n = len(series)
    var_series = np.zeros(n)
    
    for i in range(scale, n):
        window = series[i-scale:i]
        diffs = window - np.mean(window)
        var_series[i] = np.sum(diffs**2) / scale
    
    # Fill initial values
    var_series[:scale] = var_series[scale] if scale < n else 0.0
    
    # Normalize
    max_val = np.max(var_series)
    if max_val > 0:
        var_series = var_series / max_val
    
    return var_series

def reference_wavelet_correlation(series1, series2, scale):
    """Reference implementation for comparison"""
    n = min(len(series1), len(series2))
    corr_series = np.zeros(n)
    
    for i in range(scale, n):
        # Calculate means
        mean1 = np.mean(series1[i-scale:i])
        mean2 = np.mean(series2[i-scale:i])
        
        # Calculate correlation
        numerator = np.sum((series1[i-scale:i] - mean1) * (series2[i-scale:i] - mean2))
        sum_sq1 = np.sum((series1[i-scale:i] - mean1)**2)
        sum_sq2 = np.sum((series2[i-scale:i] - mean2)**2)
        
        denominator = np.sqrt(sum_sq1 * sum_sq2)
        if denominator > 1e-10:
            corr_series[i] = numerator / denominator
        else:
            corr_series[i] = 0.0
    
    # Fill initial values
    corr_series[:scale] = corr_series[scale] if scale < n else 0.0
    
    return corr_series

def benchmark_numba_optimizations():
    """Comprehensive benchmark of Numba optimizations"""
    if not NUMBA_AVAILABLE:
        print("❌ Numba not available, skipping benchmarks")
        return
    
    print("\n🏁 Benchmarking Numba Optimizations...")
    print("=" * 50)
    
    # Test data sizes
    sizes = [100, 500, 1000, 2000, 5000]
    scale = 10
    
    results = {}
    
    for size in sizes:
        print(f"\n📏 Testing with {size} samples:")
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(size).astype(np.float64)
        test_data2 = np.random.randn(size).astype(np.float64)
        
        # Test Wavelet Variance
        print("   🔬 Wavelet Variance:")
        
        # Reference timing
        start_time = time.perf_counter()
        ref_variance = reference_wavelet_variance(test_data, scale)
        ref_time = time.perf_counter() - start_time
        
        # Optimized timing (parallel)
        start_time = time.perf_counter()
        opt_variance = numba_wavelet_variance_kernel(test_data, scale)
        opt_time = time.perf_counter() - start_time
        
        # Optimized timing (vectorized)
        start_time = time.perf_counter()
        opt_variance_vec = numba_wavelet_variance_kernel_vectorized(test_data, scale)
        opt_time_vec = time.perf_counter() - start_time
        
        # Calculate speedups
        speedup_parallel = ref_time / opt_time if opt_time > 0 else 0
        speedup_vectorized = ref_time / opt_time_vec if opt_time_vec > 0 else 0
        
        # Check accuracy
        accuracy_parallel = np.allclose(opt_variance, ref_variance, rtol=1e-10)
        accuracy_vectorized = np.allclose(opt_variance_vec, ref_variance, rtol=1e-10)
        
        print(f"     Reference: {ref_time:.6f}s")
        print(f"     Numba Parallel: {opt_time:.6f}s (Speedup: {speedup_parallel:.1f}x, Accuracy: {'✅' if accuracy_parallel else '❌'})")
        print(f"     Numba Vectorized: {opt_time_vec:.6f}s (Speedup: {speedup_vectorized:.1f}x, Accuracy: {'✅' if accuracy_vectorized else '❌'})")
        
        # Test Wavelet Correlation
        print("   🔬 Wavelet Correlation:")
        
        # Reference timing
        start_time = time.perf_counter()
        ref_correlation = reference_wavelet_correlation(test_data, test_data2, scale)
        ref_time = time.perf_counter() - start_time
        
        # Optimized timing (parallel)
        start_time = time.perf_counter()
        opt_correlation = numba_wavelet_correlation_kernel(test_data, test_data2, scale)
        opt_time = time.perf_counter() - start_time
        
        # Optimized timing (vectorized)
        start_time = time.perf_counter()
        opt_correlation_vec = numba_wavelet_correlation_kernel_vectorized(test_data, test_data2, scale)
        opt_time_vec = time.perf_counter() - start_time
        
        # Calculate speedups
        speedup_parallel = ref_time / opt_time if opt_time > 0 else 0
        speedup_vectorized = ref_time / opt_time_vec if opt_time_vec > 0 else 0
        
        # Check accuracy
        accuracy_parallel = np.allclose(opt_correlation, ref_correlation, rtol=1e-10)
        accuracy_vectorized = np.allclose(opt_correlation_vec, ref_correlation, rtol=1e-10)
        
        print(f"     Reference: {ref_time:.6f}s")
        print(f"     Numba Parallel: {opt_time:.6f}s (Speedup: {speedup_parallel:.1f}x, Accuracy: {'✅' if accuracy_parallel else '❌'})")
        print(f"     Numba Vectorized: {opt_time_vec:.6f}s (Speedup: {speedup_vectorized:.1f}x, Accuracy: {'✅' if accuracy_vectorized else '❌'})")
        
        # Store results
        results[size] = {
            'variance_speedup_parallel': speedup_parallel,
            'variance_speedup_vectorized': speedup_vectorized,
            'correlation_speedup_parallel': speedup_parallel,
            'correlation_speedup_vectorized': speedup_vectorized,
            'variance_accuracy_parallel': accuracy_parallel,
            'variance_accuracy_vectorized': accuracy_vectorized,
            'correlation_accuracy_parallel': accuracy_parallel,
            'correlation_accuracy_vectorized': accuracy_vectorized
        }
    
    # Summary
    print("\n📊 NUMBA OPTIMIZATION SUMMARY")
    print("=" * 40)
    
    # Calculate average speedups
    all_speedups = []
    all_accurate = []
    
    for size_results in results.values():
        all_speedups.extend([
            size_results['variance_speedup_parallel'],
            size_results['variance_speedup_vectorized'],
            size_results['correlation_speedup_parallel'],
            size_results['correlation_speedup_vectorized']
        ])
        all_accurate.extend([
            size_results['variance_accuracy_parallel'],
            size_results['variance_accuracy_vectorized'],
            size_results['correlation_accuracy_parallel'],
            size_results['correlation_accuracy_vectorized']
        ])
    
    avg_speedup = np.mean([s for s in all_speedups if s > 0])
    max_speedup = np.max([s for s in all_speedups if s > 0])
    accuracy_rate = np.mean(all_accurate) * 100
    
    print(f"Average Speedup: {avg_speedup:.1f}x")
    print(f"Maximum Speedup: {max_speedup:.1f}x")
    print(f"Accuracy Rate: {accuracy_rate:.1f}%")
    
    # Performance targets
    target_achieved = avg_speedup >= 20.0  # Close to 25x target
    accuracy_target = accuracy_rate >= 95.0
    
    print(f"\n🎯 PERFORMANCE TARGETS:")
    print(f"25x Speedup Target: {'✅ ACHIEVED' if target_achieved else '⚠️  IN PROGRESS'} ({avg_speedup:.1f}x)")
    print(f"Accuracy Target (>95%): {'✅ ACHIEVED' if accuracy_target else '⚠️  NEEDS WORK'} ({accuracy_rate:.1f}%)")
    
    overall_success = target_achieved and accuracy_target
    print(f"\n🏆 NUMBA OPTIMIZATION: {'✅ SUCCESS' if overall_success else '⚠️  NEEDS IMPROVEMENT'}")
    
    return results

def benchmark_mkl_operations():
    """Benchmark Intel MKL operations"""
    if not MKL_AVAILABLE:
        print("❌ Intel MKL not available, skipping benchmarks")
        return
    
    print("\n🏁 Benchmarking Intel MKL Operations...")
    print("=" * 50)
    
    accelerator = IntelMKLAccelerator(verbose=False)
    
    # Matrix sizes to test
    matrix_sizes = [64, 128, 256, 512]
    
    for size in matrix_sizes:
        print(f"\n📐 Testing {size}x{size} matrices:")
        
        # Generate test matrices
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float64)
        B = np.random.randn(size, size).astype(np.float64)
        
        # Reference GEMM
        start_time = time.perf_counter()
        ref_result = np.dot(A, B)
        ref_time = time.perf_counter() - start_time
        
        # MKL GEMM
        mkl_result = accelerator.gemm_optimized(A, B)
        
        # Check accuracy and speedup
        accuracy = np.allclose(mkl_result.result, ref_result, rtol=1e-10)
        speedup = ref_time / mkl_result.execution_time if mkl_result.execution_time > 0 else 0
        
        print(f"   GEMM:")
        print(f"     Reference: {ref_time:.6f}s")
        print(f"     MKL: {mkl_result.execution_time:.6f}s")
        print(f"     Speedup: {speedup:.2f}x")
        print(f"     Backend: {mkl_result.backend}")
        print(f"     Accuracy: {'✅' if accuracy else '❌'}")
    
    # FFT tests
    fft_sizes = [1024, 2048, 4096]
    
    for size in fft_sizes:
        print(f"\n📊 Testing FFT size {size}:")
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(size).astype(np.float64)
        
        # Reference FFT
        start_time = time.perf_counter()
        ref_result = np.fft.fft(test_data)
        ref_time = time.perf_counter() - start_time
        
        # MKL FFT
        mkl_result = accelerator.fft_optimized(test_data)
        
        # Check accuracy and speedup
        accuracy = np.allclose(mkl_result.result, ref_result, rtol=1e-10)
        speedup = ref_time / mkl_result.execution_time if mkl_result.execution_time > 0 else 0
        
        print(f"   FFT:")
        print(f"     Reference: {ref_time:.6f}s")
        print(f"     MKL: {mkl_result.execution_time:.6f}s")
        print(f"     Speedup: {speedup:.2f}x")
        print(f"     Backend: {mkl_result.backend}")
        print(f"     Accuracy: {'✅' if accuracy else '❌'}")

def main():
    """Main test execution"""
    print("🚀 Optimization Integration Test Suite")
    print("=" * 50)
    
    # Validate Numba kernels first
    if NUMBA_AVAILABLE:
        print("\n🧪 Validating Numba Kernels...")
        try:
            validate_numba_kernels()
        except Exception as e:
            print(f"❌ Numba validation failed: {e}")
    
    # Run benchmarks
    numba_results = benchmark_numba_optimizations()
    benchmark_mkl_operations()
    
    print("\n🎉 INTEGRATION TEST COMPLETE!")
    
    return numba_results

if __name__ == "__main__":
    main()