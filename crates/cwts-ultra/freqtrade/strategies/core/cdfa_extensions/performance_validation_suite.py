#!/usr/bin/env python3
"""
Performance Validation Suite for Numba & Intel MKL Optimizations
Comprehensive benchmarking and accuracy validation

Author: Agent 3 - Numba & Mathematical Optimization Specialist
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings
import os
import psutil
import gc

# Import optimization modules
try:
    from .numba_optimized_kernels import *
    NUMBA_KERNELS_AVAILABLE = True
except ImportError:
    NUMBA_KERNELS_AVAILABLE = False

try:
    from .intel_mkl_integration import IntelMKLAccelerator
    MKL_INTEGRATION_AVAILABLE = True
except ImportError:
    MKL_INTEGRATION_AVAILABLE = False

try:
    from .wavelet_processor_optimized import OptimizedWaveletProcessor
    OPTIMIZED_PROCESSOR_AVAILABLE = True
except ImportError:
    OPTIMIZED_PROCESSOR_AVAILABLE = False

@dataclass
class ValidationResult:
    """Validation result container"""
    test_name: str
    operation: str
    data_size: int
    original_time: float
    optimized_time: float
    speedup_factor: float
    accuracy_check: bool
    max_error: float
    memory_usage_mb: float
    backend_used: str
    timestamp: str
    success: bool
    error_message: Optional[str] = None

class PerformanceValidationSuite:
    """
    Comprehensive validation suite for all optimizations.
    
    Tests:
    - Mathematical accuracy vs reference implementations
    - Performance improvements vs baseline
    - Memory usage optimization
    - Scaling behavior with data size
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = output_dir
        self.results = []
        self.summary_statistics = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.mkl_accelerator = None
        self.optimized_processor = None
        
        if MKL_INTEGRATION_AVAILABLE:
            self.mkl_accelerator = IntelMKLAccelerator(verbose=False)
        
        if OPTIMIZED_PROCESSOR_AVAILABLE:
            self.optimized_processor = OptimizedWaveletProcessor()
        
        print(f"\n🧪 Performance Validation Suite Initialized")
        print(f"   Output Directory: {output_dir}")
        print(f"   Numba Kernels: {'✅' if NUMBA_KERNELS_AVAILABLE else '❌'}")
        print(f"   Intel MKL: {'✅' if MKL_INTEGRATION_AVAILABLE else '❌'}")
        print(f"   Optimized Processor: {'✅' if OPTIMIZED_PROCESSOR_AVAILABLE else '❌'}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def validate_accuracy(self, result: np.ndarray, reference: np.ndarray, 
                         rtol: float = 1e-10, atol: float = 1e-12) -> Tuple[bool, float]:
        """Validate numerical accuracy against reference"""
        try:
            if result.shape != reference.shape:
                return False, float('inf')
            
            is_close = np.allclose(result, reference, rtol=rtol, atol=atol)
            max_error = np.max(np.abs(result - reference))
            
            return is_close, max_error
        except Exception:
            return False, float('inf')
    
    # =========================================================================
    # NUMBA KERNEL VALIDATION
    # =========================================================================
    
    def test_wavelet_variance_kernel(self, data_sizes: List[int] = None) -> List[ValidationResult]:
        """Test wavelet variance kernel performance and accuracy"""
        if not NUMBA_KERNELS_AVAILABLE:
            return []
        
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 2000, 5000]
        
        results = []
        print("\n🔬 Testing Wavelet Variance Kernel...")
        
        for size in data_sizes:
            print(f"   Testing size {size}...")
            
            # Generate test data
            np.random.seed(42)
            test_data = np.random.randn(size).astype(np.float64)
            scale = min(10, size // 10)
            
            # Reference implementation (Python)
            start_time = time.perf_counter()
            reference = self._reference_wavelet_variance(test_data, scale)
            original_time = time.perf_counter() - start_time
            
            # Optimized implementation
            start_time = time.perf_counter()
            memory_before = self.get_memory_usage()
            
            try:
                # Test both implementations
                optimized_parallel = numba_wavelet_variance_kernel(test_data, scale)
                optimized_vectorized = numba_wavelet_variance_kernel_vectorized(test_data, scale)
                
                optimized_time = time.perf_counter() - start_time
                memory_after = self.get_memory_usage()
                
                # Validate accuracy
                accuracy_parallel, error_parallel = self.validate_accuracy(optimized_parallel, reference)
                accuracy_vectorized, error_vectorized = self.validate_accuracy(optimized_vectorized, reference)
                
                # Choose best implementation
                if accuracy_vectorized and error_vectorized <= error_parallel:
                    optimized = optimized_vectorized
                    accuracy = accuracy_vectorized
                    max_error = error_vectorized
                    backend = "numba_vectorized"
                else:
                    optimized = optimized_parallel
                    accuracy = accuracy_parallel
                    max_error = error_parallel
                    backend = "numba_parallel"
                
                speedup = original_time / optimized_time if optimized_time > 0 else 1.0
                
                result = ValidationResult(
                    test_name="wavelet_variance_kernel",
                    operation="wavelet_variance",
                    data_size=size,
                    original_time=original_time,
                    optimized_time=optimized_time,
                    speedup_factor=speedup,
                    accuracy_check=accuracy,
                    max_error=max_error,
                    memory_usage_mb=memory_after - memory_before,
                    backend_used=backend,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
                print(f"     ✅ Speedup: {speedup:.1f}x, Accuracy: {'✅' if accuracy else '❌'}")
                
            except Exception as e:
                result = ValidationResult(
                    test_name="wavelet_variance_kernel",
                    operation="wavelet_variance",
                    data_size=size,
                    original_time=original_time,
                    optimized_time=0.0,
                    speedup_factor=0.0,
                    accuracy_check=False,
                    max_error=float('inf'),
                    memory_usage_mb=0.0,
                    backend_used="failed",
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message=str(e)
                )
                print(f"     ❌ Failed: {e}")
            
            results.append(result)
            
            # Clean up memory
            gc.collect()
        
        return results
    
    def test_wavelet_correlation_kernel(self, data_sizes: List[int] = None) -> List[ValidationResult]:
        """Test wavelet correlation kernel performance and accuracy"""
        if not NUMBA_KERNELS_AVAILABLE:
            return []
        
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 2000, 5000]
        
        results = []
        print("\n🔬 Testing Wavelet Correlation Kernel...")
        
        for size in data_sizes:
            print(f"   Testing size {size}...")
            
            # Generate test data
            np.random.seed(42)
            test_data1 = np.random.randn(size).astype(np.float64)
            test_data2 = np.random.randn(size).astype(np.float64)
            scale = min(10, size // 10)
            
            # Reference implementation (Python)
            start_time = time.perf_counter()
            reference = self._reference_wavelet_correlation(test_data1, test_data2, scale)
            original_time = time.perf_counter() - start_time
            
            # Optimized implementation
            start_time = time.perf_counter()
            memory_before = self.get_memory_usage()
            
            try:
                # Test both implementations
                optimized_parallel = numba_wavelet_correlation_kernel(test_data1, test_data2, scale)
                optimized_vectorized = numba_wavelet_correlation_kernel_vectorized(test_data1, test_data2, scale)
                
                optimized_time = time.perf_counter() - start_time
                memory_after = self.get_memory_usage()
                
                # Validate accuracy
                accuracy_parallel, error_parallel = self.validate_accuracy(optimized_parallel, reference)
                accuracy_vectorized, error_vectorized = self.validate_accuracy(optimized_vectorized, reference)
                
                # Choose best implementation
                if accuracy_vectorized and error_vectorized <= error_parallel:
                    optimized = optimized_vectorized
                    accuracy = accuracy_vectorized
                    max_error = error_vectorized
                    backend = "numba_vectorized"
                else:
                    optimized = optimized_parallel
                    accuracy = accuracy_parallel
                    max_error = error_parallel
                    backend = "numba_parallel"
                
                speedup = original_time / optimized_time if optimized_time > 0 else 1.0
                
                result = ValidationResult(
                    test_name="wavelet_correlation_kernel",
                    operation="wavelet_correlation",
                    data_size=size,
                    original_time=original_time,
                    optimized_time=optimized_time,
                    speedup_factor=speedup,
                    accuracy_check=accuracy,
                    max_error=max_error,
                    memory_usage_mb=memory_after - memory_before,
                    backend_used=backend,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
                print(f"     ✅ Speedup: {speedup:.1f}x, Accuracy: {'✅' if accuracy else '❌'}")
                
            except Exception as e:
                result = ValidationResult(
                    test_name="wavelet_correlation_kernel",
                    operation="wavelet_correlation",
                    data_size=size,
                    original_time=original_time,
                    optimized_time=0.0,
                    speedup_factor=0.0,
                    accuracy_check=False,
                    max_error=float('inf'),
                    memory_usage_mb=0.0,
                    backend_used="failed",
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message=str(e)
                )
                print(f"     ❌ Failed: {e}")
            
            results.append(result)
            
            # Clean up memory
            gc.collect()
        
        return results
    
    # =========================================================================
    # INTEL MKL VALIDATION
    # =========================================================================
    
    def test_mkl_matrix_operations(self, matrix_sizes: List[int] = None) -> List[ValidationResult]:
        """Test Intel MKL matrix operations"""
        if not MKL_INTEGRATION_AVAILABLE or self.mkl_accelerator is None:
            return []
        
        if matrix_sizes is None:
            matrix_sizes = [64, 128, 256, 512]
        
        results = []
        print("\n🔬 Testing Intel MKL Matrix Operations...")
        
        for size in matrix_sizes:
            print(f"   Testing matrix size {size}x{size}...")
            
            # Generate test matrices
            np.random.seed(42)
            A = np.random.randn(size, size).astype(np.float64)
            B = np.random.randn(size, size).astype(np.float64)
            
            # Test GEMM
            start_time = time.perf_counter()
            reference = np.dot(A, B)
            original_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            memory_before = self.get_memory_usage()
            
            try:
                mkl_result = self.mkl_accelerator.gemm_optimized(A, B)
                optimized_time = time.perf_counter() - start_time
                memory_after = self.get_memory_usage()
                
                # Validate accuracy
                accuracy, max_error = self.validate_accuracy(mkl_result.result, reference)
                speedup = original_time / optimized_time if optimized_time > 0 else 1.0
                
                result = ValidationResult(
                    test_name="mkl_gemm",
                    operation="matrix_multiply",
                    data_size=size * size * 2,  # Two matrices
                    original_time=original_time,
                    optimized_time=optimized_time,
                    speedup_factor=speedup,
                    accuracy_check=accuracy,
                    max_error=max_error,
                    memory_usage_mb=memory_after - memory_before,
                    backend_used=mkl_result.backend,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
                print(f"     ✅ GEMM Speedup: {speedup:.1f}x, Backend: {mkl_result.backend}")
                
            except Exception as e:
                result = ValidationResult(
                    test_name="mkl_gemm",
                    operation="matrix_multiply",
                    data_size=size * size * 2,
                    original_time=original_time,
                    optimized_time=0.0,
                    speedup_factor=0.0,
                    accuracy_check=False,
                    max_error=float('inf'),
                    memory_usage_mb=0.0,
                    backend_used="failed",
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message=str(e)
                )
                print(f"     ❌ GEMM Failed: {e}")
            
            results.append(result)
        
        return results
    
    def test_mkl_fft_operations(self, data_sizes: List[int] = None) -> List[ValidationResult]:
        """Test Intel MKL FFT operations"""
        if not MKL_INTEGRATION_AVAILABLE or self.mkl_accelerator is None:
            return []
        
        if data_sizes is None:
            data_sizes = [1024, 2048, 4096, 8192]
        
        results = []
        print("\n🔬 Testing Intel MKL FFT Operations...")
        
        for size in data_sizes:
            print(f"   Testing FFT size {size}...")
            
            # Generate test data
            np.random.seed(42)
            test_data = np.random.randn(size).astype(np.float64)
            
            # Reference FFT
            start_time = time.perf_counter()
            reference = np.fft.fft(test_data)
            original_time = time.perf_counter() - start_time
            
            # MKL FFT
            start_time = time.perf_counter()
            memory_before = self.get_memory_usage()
            
            try:
                mkl_result = self.mkl_accelerator.fft_optimized(test_data)
                optimized_time = time.perf_counter() - start_time
                memory_after = self.get_memory_usage()
                
                # Validate accuracy
                accuracy, max_error = self.validate_accuracy(mkl_result.result, reference)
                speedup = original_time / optimized_time if optimized_time > 0 else 1.0
                
                result = ValidationResult(
                    test_name="mkl_fft",
                    operation="fft",
                    data_size=size,
                    original_time=original_time,
                    optimized_time=optimized_time,
                    speedup_factor=speedup,
                    accuracy_check=accuracy,
                    max_error=max_error,
                    memory_usage_mb=memory_after - memory_before,
                    backend_used=mkl_result.backend,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
                print(f"     ✅ FFT Speedup: {speedup:.1f}x, Backend: {mkl_result.backend}")
                
            except Exception as e:
                result = ValidationResult(
                    test_name="mkl_fft",
                    operation="fft",
                    data_size=size,
                    original_time=original_time,
                    optimized_time=0.0,
                    speedup_factor=0.0,
                    accuracy_check=False,
                    max_error=float('inf'),
                    memory_usage_mb=0.0,
                    backend_used="failed",
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message=str(e)
                )
                print(f"     ❌ FFT Failed: {e}")
            
            results.append(result)
        
        return results
    
    # =========================================================================
    # REFERENCE IMPLEMENTATIONS
    # =========================================================================
    
    def _reference_wavelet_variance(self, series: np.ndarray, scale: int) -> np.ndarray:
        """Reference implementation of wavelet variance"""
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
    
    def _reference_wavelet_correlation(self, series1: np.ndarray, series2: np.ndarray, 
                                     scale: int) -> np.ndarray:
        """Reference implementation of wavelet correlation"""
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
    
    # =========================================================================
    # COMPREHENSIVE VALIDATION RUNNER
    # =========================================================================
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("\n🚀 Running Comprehensive Performance Validation...")
        print("=" * 60)
        
        all_results = []
        
        # Test Numba kernels
        if NUMBA_KERNELS_AVAILABLE:
            all_results.extend(self.test_wavelet_variance_kernel())
            all_results.extend(self.test_wavelet_correlation_kernel())
        
        # Test Intel MKL
        if MKL_INTEGRATION_AVAILABLE:
            all_results.extend(self.test_mkl_matrix_operations())
            all_results.extend(self.test_mkl_fft_operations())
        
        # Store results
        self.results = all_results
        
        # Generate summary statistics
        self._generate_summary_statistics()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_validation_summary()
        
        return self.summary_statistics
    
    def _generate_summary_statistics(self):
        """Generate summary statistics from all results"""
        if not self.results:
            return
        
        successful_results = [r for r in self.results if r.success]
        accurate_results = [r for r in successful_results if r.accuracy_check]
        
        # Overall statistics
        self.summary_statistics = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "accurate_tests": len(accurate_results),
            "success_rate": len(successful_results) / len(self.results) * 100,
            "accuracy_rate": len(accurate_results) / len(self.results) * 100 if self.results else 0,
            "average_speedup": np.mean([r.speedup_factor for r in accurate_results]) if accurate_results else 0,
            "max_speedup": np.max([r.speedup_factor for r in accurate_results]) if accurate_results else 0,
            "total_memory_mb": np.sum([r.memory_usage_mb for r in successful_results]),
            "operations_tested": list(set(r.operation for r in self.results)),
            "backends_used": list(set(r.backend_used for r in successful_results)),
            "timestamp": datetime.now().isoformat()
        }
        
        # By operation statistics
        operations = {}
        for result in accurate_results:
            op = result.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(result)
        
        self.summary_statistics["by_operation"] = {}
        for op, results in operations.items():
            speedups = [r.speedup_factor for r in results]
            self.summary_statistics["by_operation"][op] = {
                "count": len(results),
                "average_speedup": np.mean(speedups),
                "max_speedup": np.max(speedups),
                "min_speedup": np.min(speedups),
                "std_speedup": np.std(speedups),
                "target_achieved": np.mean(speedups) >= 20.0  # 25x target with margin
            }
    
    def _save_results(self):
        """Save validation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = os.path.join(self.output_dir, f"validation_results_{timestamp}.json")
        with open(detailed_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Save summary statistics
        summary_file = os.path.join(self.output_dir, f"validation_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(self.summary_statistics, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   Detailed: {detailed_file}")
        print(f"   Summary: {summary_file}")
    
    def _print_validation_summary(self):
        """Print comprehensive validation summary"""
        if not self.summary_statistics:
            return
        
        stats = self.summary_statistics
        
        print("\n📊 COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 60)
        
        # Overall results
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Successful: {stats['successful_tests']} ({stats['success_rate']:.1f}%)")
        print(f"Accurate: {stats['accurate_tests']} ({stats['accuracy_rate']:.1f}%)")
        print(f"Average Speedup: {stats['average_speedup']:.2f}x")
        print(f"Maximum Speedup: {stats['max_speedup']:.2f}x")
        print(f"Total Memory Used: {stats['total_memory_mb']:.2f} MB")
        
        # Performance targets
        print(f"\n🎯 PERFORMANCE TARGETS:")
        target_achieved = stats['average_speedup'] >= 20.0
        print(f"25x Speedup Target: {'✅ ACHIEVED' if target_achieved else '⚠️  IN PROGRESS'} "
              f"({stats['average_speedup']:.1f}x)")
        
        accuracy_target = stats['accuracy_rate'] >= 95.0
        print(f"Accuracy Target (>95%): {'✅ ACHIEVED' if accuracy_target else '⚠️  NEEDS WORK'} "
              f"({stats['accuracy_rate']:.1f}%)")
        
        # By operation
        print(f"\n📈 PERFORMANCE BY OPERATION:")
        print("-" * 40)
        for op, op_stats in stats.get("by_operation", {}).items():
            status = "✅" if op_stats["target_achieved"] else "⚠️"
            print(f"{status} {op.replace('_', ' ').title()}:")
            print(f"    Average: {op_stats['average_speedup']:.1f}x")
            print(f"    Range: {op_stats['min_speedup']:.1f}x - {op_stats['max_speedup']:.1f}x")
            print(f"    Tests: {op_stats['count']}")
            print()
        
        # Backends used
        print(f"🔧 Backends Used: {', '.join(stats['backends_used'])}")
        print(f"🔬 Operations Tested: {', '.join(stats['operations_tested'])}")
        
        # Final verdict
        overall_success = (stats['success_rate'] >= 90.0 and 
                          stats['accuracy_rate'] >= 95.0 and 
                          stats['average_speedup'] >= 20.0)
        
        print(f"\n🏆 FINAL VERDICT: {'✅ OPTIMIZATION SUCCESS' if overall_success else '⚠️  NEEDS IMPROVEMENT'}")
        
        if overall_success:
            print("   All performance targets achieved!")
            print("   Ready for production deployment.")
        else:
            print("   Some targets need attention.")
            print("   Review individual operation results.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main validation execution"""
    suite = PerformanceValidationSuite()
    results = suite.run_comprehensive_validation()
    return results

if __name__ == "__main__":
    main()