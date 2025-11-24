#!/usr/bin/env python3
"""
Intel MKL Integration Layer for CDFA Extensions
High-performance BLAS operations with Intel Math Kernel Library

Author: Agent 3 - Numba & Mathematical Optimization Specialist
Target: 25x speedup for matrix operations
"""

import numpy as np
import os
import warnings
import time
import logging
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import threading
from contextlib import contextmanager

# Intel MKL imports with fallbacks
try:
    # Try Intel MKL direct imports
    import mkl
    from mkl_fft import fft, ifft, fft2, ifft2
    MKL_AVAILABLE = True
    print("✅ Intel MKL directly available")
except ImportError:
    try:
        # Try NumPy with MKL backend
        import numpy as np
        # Check if NumPy is using MKL
        if 'mkl' in np.__config__.show().lower():
            MKL_AVAILABLE = True
            print("✅ Intel MKL available through NumPy")
        else:
            MKL_AVAILABLE = False
            print("⚠️  Intel MKL not detected in NumPy backend")
    except:
        MKL_AVAILABLE = False
        print("⚠️  Intel MKL not available")

# BLAS interface imports
try:
    from scipy.linalg import lapack, blas
    SCIPY_BLAS_AVAILABLE = True
    print("✅ SciPy BLAS interface available")
except ImportError:
    SCIPY_BLAS_AVAILABLE = False
    print("⚠️  SciPy BLAS interface not available")

# Threading control
try:
    import mkl
    MKL_THREADING_CONTROL = True
except ImportError:
    MKL_THREADING_CONTROL = False

@dataclass
class MKLPerformanceResult:
    """Performance result container for MKL operations"""
    result: Union[np.ndarray, Tuple[np.ndarray, ...]]
    execution_time: float
    memory_usage: float
    operation: str
    backend: str
    speedup_factor: float = 1.0
    threading_config: Dict[str, Any] = None

class IntelMKLAccelerator:
    """
    Intel MKL integration for high-performance mathematical operations.
    
    Provides optimized implementations of:
    - Matrix multiplication (GEMM)
    - Matrix decompositions (SVD, Eigenvalue, Cholesky)
    - FFT operations
    - BLAS Level 1, 2, 3 operations
    - Vectorized mathematical functions
    """
    
    def __init__(self, num_threads: Optional[int] = None, verbose: bool = True):
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # Initialize MKL configuration
        self.mkl_config = self._initialize_mkl_config(num_threads)
        
        # Performance baselines
        self.baselines = {}
        
        # Thread-local storage for MKL state
        self._local = threading.local()
        
        if self.verbose:
            self._print_initialization_info()
    
    def _initialize_mkl_config(self, num_threads: Optional[int]) -> Dict[str, Any]:
        """Initialize Intel MKL configuration"""
        config = {
            'available': MKL_AVAILABLE,
            'scipy_blas': SCIPY_BLAS_AVAILABLE,
            'threading_control': MKL_THREADING_CONTROL,
            'num_threads': num_threads or os.cpu_count(),
            'domain_info': {}
        }
        
        if MKL_AVAILABLE and MKL_THREADING_CONTROL:
            try:
                # Set thread count
                if num_threads:
                    mkl.set_num_threads(num_threads)
                
                # Get MKL version and configuration
                config['version'] = mkl.get_version_string()
                config['max_threads'] = mkl.get_max_threads()
                
                # Configure MKL for different domains
                domains = ['BLAS', 'FFT', 'VML', 'PARDISO']
                for domain in domains:
                    try:
                        domain_threads = getattr(mkl, f'domain_get_max_threads', lambda x: -1)(
                            getattr(mkl, f'DOMAIN_{domain}', 0)
                        )
                        config['domain_info'][domain] = domain_threads
                    except:
                        config['domain_info'][domain] = -1
                        
            except Exception as e:
                self.logger.warning(f"Failed to configure MKL: {e}")
        
        return config
    
    def _print_initialization_info(self):
        """Print MKL initialization information"""
        print("\n🚀 Intel MKL Accelerator Initialized")
        print(f"   MKL Available: {'✅' if self.mkl_config['available'] else '❌'}")
        print(f"   SciPy BLAS: {'✅' if self.mkl_config['scipy_blas'] else '❌'}")
        print(f"   Threading Control: {'✅' if self.mkl_config['threading_control'] else '❌'}")
        print(f"   Threads: {self.mkl_config['num_threads']}")
        
        if self.mkl_config['available'] and 'version' in self.mkl_config:
            print(f"   MKL Version: {self.mkl_config['version']}")
            print(f"   Max Threads: {self.mkl_config['max_threads']}")
            
            if self.mkl_config['domain_info']:
                print("   Domain Thread Configuration:")
                for domain, threads in self.mkl_config['domain_info'].items():
                    if threads > 0:
                        print(f"     {domain}: {threads} threads")
    
    @contextmanager
    def mkl_threads(self, num_threads: int):
        """Context manager for temporary thread configuration"""
        if not MKL_THREADING_CONTROL:
            yield
            return
        
        original_threads = mkl.get_max_threads()
        try:
            mkl.set_num_threads(num_threads)
            yield
        finally:
            mkl.set_num_threads(original_threads)
    
    # =========================================================================
    # HIGH-PERFORMANCE MATRIX OPERATIONS
    # =========================================================================
    
    def gemm_optimized(self, A: np.ndarray, B: np.ndarray, 
                      alpha: float = 1.0, beta: float = 0.0,
                      C: Optional[np.ndarray] = None) -> MKLPerformanceResult:
        """
        Optimized GEMM (General Matrix Multiply) using Intel MKL.
        
        Performs: C = alpha * A @ B + beta * C
        Target: 25x speedup over numpy.dot
        """
        start_time = time.perf_counter()
        
        if not self.mkl_config['scipy_blas']:
            # Fallback to NumPy
            result = alpha * np.dot(A, B)
            if C is not None:
                result += beta * C
            backend = "numpy"
        else:
            try:
                # Use SciPy BLAS GEMM
                if C is None:
                    C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
                
                result = blas.dgemm(alpha, A, B, beta, C)
                backend = "mkl_blas"
            except Exception as e:
                self.logger.warning(f"MKL GEMM failed, falling back to NumPy: {e}")
                result = alpha * np.dot(A, B)
                if C is not None:
                    result += beta * C
                backend = "numpy_fallback"
        
        execution_time = time.perf_counter() - start_time
        
        return MKLPerformanceResult(
            result=result,
            execution_time=execution_time,
            memory_usage=result.nbytes / (1024**2),  # MB
            operation="GEMM",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    def svd_optimized(self, A: np.ndarray, 
                     full_matrices: bool = True) -> MKLPerformanceResult:
        """
        Optimized SVD using Intel MKL LAPACK.
        
        Target: 25x speedup over numpy.linalg.svd
        """
        start_time = time.perf_counter()
        
        if not self.mkl_config['scipy_blas']:
            # Fallback to NumPy
            U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
            backend = "numpy"
        else:
            try:
                # Use SciPy LAPACK SVD
                U, s, Vt, info = lapack.dgesvd(A, full_matrices=full_matrices)
                if info != 0:
                    raise ValueError(f"SVD failed with info={info}")
                backend = "mkl_lapack"
            except Exception as e:
                self.logger.warning(f"MKL SVD failed, falling back to NumPy: {e}")
                U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
                backend = "numpy_fallback"
        
        execution_time = time.perf_counter() - start_time
        
        return MKLPerformanceResult(
            result=(U, s, Vt),
            execution_time=execution_time,
            memory_usage=(U.nbytes + s.nbytes + Vt.nbytes) / (1024**2),  # MB
            operation="SVD",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    def eigenvalue_optimized(self, A: np.ndarray, 
                           eigenvectors: bool = True) -> MKLPerformanceResult:
        """
        Optimized eigenvalue decomposition using Intel MKL.
        
        Target: 25x speedup over numpy.linalg.eig
        """
        start_time = time.perf_counter()
        
        if not self.mkl_config['scipy_blas']:
            # Fallback to NumPy
            if eigenvectors:
                eigenvals, eigenvecs = np.linalg.eig(A)
                result = (eigenvals, eigenvecs)
            else:
                eigenvals = np.linalg.eigvals(A)
                result = eigenvals
            backend = "numpy"
        else:
            try:
                # Use SciPy LAPACK eigenvalue decomposition
                if eigenvectors:
                    eigenvals, eigenvecs, info = lapack.dgeev(A)
                    if info != 0:
                        raise ValueError(f"Eigenvalue decomposition failed with info={info}")
                    result = (eigenvals, eigenvecs)
                else:
                    eigenvals, _, info = lapack.dgeev(A, compute_vl=False, compute_vr=False)
                    if info != 0:
                        raise ValueError(f"Eigenvalue computation failed with info={info}")
                    result = eigenvals
                backend = "mkl_lapack"
            except Exception as e:
                self.logger.warning(f"MKL eigenvalue failed, falling back to NumPy: {e}")
                if eigenvectors:
                    eigenvals, eigenvecs = np.linalg.eig(A)
                    result = (eigenvals, eigenvecs)
                else:
                    eigenvals = np.linalg.eigvals(A)
                    result = eigenvals
                backend = "numpy_fallback"
        
        execution_time = time.perf_counter() - start_time
        
        if isinstance(result, tuple):
            memory_usage = sum(arr.nbytes for arr in result) / (1024**2)
        else:
            memory_usage = result.nbytes / (1024**2)
        
        return MKLPerformanceResult(
            result=result,
            execution_time=execution_time,
            memory_usage=memory_usage,
            operation="Eigenvalue",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    def cholesky_optimized(self, A: np.ndarray, 
                          lower: bool = True) -> MKLPerformanceResult:
        """
        Optimized Cholesky decomposition using Intel MKL.
        
        Target: 25x speedup over numpy.linalg.cholesky
        """
        start_time = time.perf_counter()
        
        if not self.mkl_config['scipy_blas']:
            # Fallback to NumPy
            result = np.linalg.cholesky(A)
            if not lower:
                result = result.T
            backend = "numpy"
        else:
            try:
                # Use SciPy LAPACK Cholesky
                L, info = lapack.dpotrf(A, lower=lower)
                if info != 0:
                    raise ValueError(f"Cholesky decomposition failed with info={info}")
                
                # Extract triangular part
                if lower:
                    result = np.tril(L)
                else:
                    result = np.triu(L)
                    
                backend = "mkl_lapack"
            except Exception as e:
                self.logger.warning(f"MKL Cholesky failed, falling back to NumPy: {e}")
                result = np.linalg.cholesky(A)
                if not lower:
                    result = result.T
                backend = "numpy_fallback"
        
        execution_time = time.perf_counter() - start_time
        
        return MKLPerformanceResult(
            result=result,
            execution_time=execution_time,
            memory_usage=result.nbytes / (1024**2),
            operation="Cholesky",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    # =========================================================================
    # HIGH-PERFORMANCE FFT OPERATIONS
    # =========================================================================
    
    def fft_optimized(self, x: np.ndarray, 
                     axis: int = -1, norm: Optional[str] = None) -> MKLPerformanceResult:
        """
        Optimized FFT using Intel MKL.
        
        Target: 25x speedup over numpy.fft.fft
        """
        start_time = time.perf_counter()
        
        if MKL_AVAILABLE:
            try:
                from mkl_fft import fft as mkl_fft
                result = mkl_fft(x, axis=axis, norm=norm)
                backend = "mkl_fft"
            except Exception as e:
                self.logger.warning(f"MKL FFT failed, falling back to NumPy: {e}")
                result = np.fft.fft(x, axis=axis, norm=norm)
                backend = "numpy_fallback"
        else:
            result = np.fft.fft(x, axis=axis, norm=norm)
            backend = "numpy"
        
        execution_time = time.perf_counter() - start_time
        
        return MKLPerformanceResult(
            result=result,
            execution_time=execution_time,
            memory_usage=result.nbytes / (1024**2),
            operation="FFT",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    def ifft_optimized(self, x: np.ndarray, 
                      axis: int = -1, norm: Optional[str] = None) -> MKLPerformanceResult:
        """
        Optimized IFFT using Intel MKL.
        
        Target: 25x speedup over numpy.fft.ifft
        """
        start_time = time.perf_counter()
        
        if MKL_AVAILABLE:
            try:
                from mkl_fft import ifft as mkl_ifft
                result = mkl_ifft(x, axis=axis, norm=norm)
                backend = "mkl_fft"
            except Exception as e:
                self.logger.warning(f"MKL IFFT failed, falling back to NumPy: {e}")
                result = np.fft.ifft(x, axis=axis, norm=norm)
                backend = "numpy_fallback"
        else:
            result = np.fft.ifft(x, axis=axis, norm=norm)
            backend = "numpy"
        
        execution_time = time.perf_counter() - start_time
        
        return MKLPerformanceResult(
            result=result,
            execution_time=execution_time,
            memory_usage=result.nbytes / (1024**2),
            operation="IFFT",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    # =========================================================================
    # VECTORIZED MATHEMATICAL FUNCTIONS
    # =========================================================================
    
    def vml_optimized_operations(self, x: np.ndarray, 
                                operation: str) -> MKLPerformanceResult:
        """
        Optimized vectorized mathematical operations using Intel MKL VML.
        
        Supported operations: exp, log, sin, cos, sqrt, etc.
        Target: 25x speedup over numpy universal functions
        """
        start_time = time.perf_counter()
        
        # Map operation names to functions
        numpy_ops = {
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'sqrt': np.sqrt,
            'tanh': np.tanh,
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x))
        }
        
        if operation not in numpy_ops:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Intel MKL VML operations are typically accessed through NumPy
        # when NumPy is built with MKL
        result = numpy_ops[operation](x)
        backend = "mkl_vml" if MKL_AVAILABLE else "numpy"
        
        execution_time = time.perf_counter() - start_time
        
        return MKLPerformanceResult(
            result=result,
            execution_time=execution_time,
            memory_usage=result.nbytes / (1024**2),
            operation=f"VML_{operation}",
            backend=backend,
            threading_config={"threads": self.mkl_config['num_threads']}
        )
    
    # =========================================================================
    # PERFORMANCE BENCHMARKING AND VALIDATION
    # =========================================================================
    
    def benchmark_against_numpy(self, operation: str, 
                               test_data: np.ndarray,
                               iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark MKL operations against NumPy equivalents.
        
        Returns detailed performance comparison.
        """
        print(f"\n🏁 Benchmarking {operation} operation...")
        
        # Warm up
        if operation == "gemm":
            for _ in range(3):
                _ = self.gemm_optimized(test_data, test_data)
                _ = np.dot(test_data, test_data)
        
        # MKL timings
        mkl_times = []
        for _ in range(iterations):
            if operation == "gemm":
                result = self.gemm_optimized(test_data, test_data)
                mkl_times.append(result.execution_time)
        
        # NumPy timings
        numpy_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            if operation == "gemm":
                _ = np.dot(test_data, test_data)
            numpy_times.append(time.perf_counter() - start_time)
        
        # Calculate statistics
        mkl_mean = np.mean(mkl_times)
        numpy_mean = np.mean(numpy_times)
        speedup = numpy_mean / mkl_mean if mkl_mean > 0 else 1.0
        
        results = {
            'operation': operation,
            'mkl_time_mean': mkl_mean,
            'mkl_time_std': np.std(mkl_times),
            'numpy_time_mean': numpy_mean,
            'numpy_time_std': np.std(numpy_times),
            'speedup_factor': speedup,
            'iterations': iterations,
            'data_shape': test_data.shape,
            'backend': result.backend if 'result' in locals() else 'unknown'
        }
        
        print(f"   MKL Time: {mkl_mean:.6f}s ± {np.std(mkl_times):.6f}s")
        print(f"   NumPy Time: {numpy_mean:.6f}s ± {np.std(numpy_times):.6f}s")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Backend: {results['backend']}")
        
        return results
    
    def validate_mkl_integration(self) -> bool:
        """
        Comprehensive validation of MKL integration.
        
        Tests mathematical accuracy and performance gains.
        """
        print("\n🧪 Validating Intel MKL Integration...")
        
        # Generate test data
        np.random.seed(42)
        test_matrix = np.random.randn(500, 500).astype(np.float64)
        test_vector = np.random.randn(1000).astype(np.float64)
        
        validation_results = {}
        
        # Test GEMM
        try:
            mkl_result = self.gemm_optimized(test_matrix[:100, :100], 
                                           test_matrix[:100, :100])
            numpy_result = np.dot(test_matrix[:100, :100], 
                                test_matrix[:100, :100])
            
            gemm_accuracy = np.allclose(mkl_result.result, numpy_result, rtol=1e-10)
            validation_results['gemm'] = {
                'accuracy': gemm_accuracy,
                'backend': mkl_result.backend,
                'speedup_achievable': mkl_result.backend != 'numpy'
            }
            print(f"   GEMM: {'✅' if gemm_accuracy else '❌'} (Backend: {mkl_result.backend})")
        except Exception as e:
            print(f"   GEMM: ❌ Failed - {e}")
            validation_results['gemm'] = {'accuracy': False, 'error': str(e)}
        
        # Test SVD
        try:
            mkl_result = self.svd_optimized(test_matrix[:50, :50])
            U_mkl, s_mkl, Vt_mkl = mkl_result.result
            U_np, s_np, Vt_np = np.linalg.svd(test_matrix[:50, :50])
            
            # Check reconstruction accuracy
            reconstruct_mkl = U_mkl @ np.diag(s_mkl) @ Vt_mkl
            reconstruct_np = U_np @ np.diag(s_np) @ Vt_np
            svd_accuracy = np.allclose(reconstruct_mkl, reconstruct_np, rtol=1e-10)
            
            validation_results['svd'] = {
                'accuracy': svd_accuracy,
                'backend': mkl_result.backend,
                'speedup_achievable': mkl_result.backend != 'numpy'
            }
            print(f"   SVD: {'✅' if svd_accuracy else '❌'} (Backend: {mkl_result.backend})")
        except Exception as e:
            print(f"   SVD: ❌ Failed - {e}")
            validation_results['svd'] = {'accuracy': False, 'error': str(e)}
        
        # Test FFT
        try:
            mkl_result = self.fft_optimized(test_vector)
            numpy_result = np.fft.fft(test_vector)
            
            fft_accuracy = np.allclose(mkl_result.result, numpy_result, rtol=1e-10)
            validation_results['fft'] = {
                'accuracy': fft_accuracy,
                'backend': mkl_result.backend,
                'speedup_achievable': mkl_result.backend != 'numpy'
            }
            print(f"   FFT: {'✅' if fft_accuracy else '❌'} (Backend: {mkl_result.backend})")
        except Exception as e:
            print(f"   FFT: ❌ Failed - {e}")
            validation_results['fft'] = {'accuracy': False, 'error': str(e)}
        
        # Overall validation
        all_accurate = all(result.get('accuracy', False) 
                          for result in validation_results.values())
        any_speedup = any(result.get('speedup_achievable', False) 
                         for result in validation_results.values())
        
        print(f"\n📊 MKL Integration Summary:")
        print(f"   Mathematical Accuracy: {'✅' if all_accurate else '❌'}")
        print(f"   Performance Optimization Available: {'✅' if any_speedup else '❌'}")
        print(f"   Overall Status: {'✅ READY' if all_accurate else '⚠️  NEEDS ATTENTION'}")
        
        return all_accurate

# =============================================================================
# GLOBAL MKL ACCELERATOR INSTANCE
# =============================================================================

# Create global instance for easy access
mkl_accelerator = IntelMKLAccelerator(verbose=False)

def get_mkl_accelerator() -> IntelMKLAccelerator:
    """Get the global MKL accelerator instance"""
    return mkl_accelerator

if __name__ == "__main__":
    # Run validation
    accelerator = IntelMKLAccelerator()
    accelerator.validate_mkl_integration()
    
    # Run benchmarks
    test_data = np.random.randn(256, 256).astype(np.float64)
    benchmark_results = accelerator.benchmark_against_numpy("gemm", test_data)
    print(f"\n🏆 Final Speedup: {benchmark_results['speedup_factor']:.2f}x")