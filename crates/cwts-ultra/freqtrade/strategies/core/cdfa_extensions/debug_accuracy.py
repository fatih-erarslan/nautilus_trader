#!/usr/bin/env python3
"""
Debug Accuracy Issues
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from numba_optimized_kernels import (
    numba_wavelet_variance_kernel,
    numba_wavelet_variance_kernel_vectorized
)

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
    print(f"Debug: max_val before normalization = {max_val}")
    if max_val > 0:
        var_series = var_series / max_val
    print(f"Debug: var_series after normalization = {var_series[:5]}")
    
    return var_series

def reference_wavelet_variance_raw(series, scale):
    """Reference implementation WITHOUT normalization"""
    n = len(series)
    var_series = np.zeros(n)
    
    for i in range(scale, n):
        window = series[i-scale:i]
        diffs = window - np.mean(window)
        var_series[i] = np.sum(diffs**2) / scale
    
    # Fill initial values
    var_series[:scale] = var_series[scale] if scale < n else 0.0
    
    # NO normalization
    print(f"Debug: Reference raw max_val = {np.max(var_series)}")
    print(f"Debug: Reference raw var_series = {var_series[:5]}")
    
    return var_series

def debug_variance_implementation():
    """Debug variance implementation step by step"""
    
    # Simple test case
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    scale = 3
    
    print("🔍 Debugging Variance Implementation")
    print(f"Test data: {test_data}")
    print(f"Scale: {scale}")
    
    # Reference implementation
    ref_result = reference_wavelet_variance(test_data, scale)
    print(f"\nReference result: {ref_result}")
    
    # Numba implementation  
    numba_result = numba_wavelet_variance_kernel_vectorized(test_data, scale)
    print(f"Numba result: {numba_result}")
    
    # Check the raw calculation without normalization
    print("\n🔬 Checking raw calculations without normalization:")
    numba_raw = np.copy(numba_result)
    ref_raw = reference_wavelet_variance_raw(test_data, scale)
    print(f"Reference raw: {ref_raw}")
    print(f"Numba raw: {numba_raw}")
    print(f"Raw match: {np.allclose(ref_raw, numba_raw, rtol=1e-10)}")
    
    # Check differences
    diff = np.abs(ref_result - numba_result)
    max_diff = np.max(diff)
    print(f"\nMax difference: {max_diff}")
    print(f"All close (1e-10): {np.allclose(ref_result, numba_result, rtol=1e-10)}")
    print(f"All close (1e-8): {np.allclose(ref_result, numba_result, rtol=1e-8)}")
    print(f"All close (1e-6): {np.allclose(ref_result, numba_result, rtol=1e-6)}")
    
    # Manual step-by-step comparison
    print("\n🔬 Step-by-step comparison:")
    n = len(test_data)
    for i in range(scale, min(scale + 5, n)):  # Check first few calculations
        print(f"\nIndex {i}:")
        
        # Reference calculation
        window = test_data[i-scale:i]
        mean_ref = np.mean(window)
        diffs = window - mean_ref
        variance_ref = np.sum(diffs**2) / scale
        print(f"  Reference: window={window}, mean={mean_ref:.6f}, variance={variance_ref:.6f}")
        
        # Manual Numba calculation
        window_sum = 0.0
        for j in range(scale):
            window_sum += test_data[i - j]
        mean_numba = window_sum / scale
        
        variance_numba = 0.0
        for j in range(scale):
            diff = test_data[i - j] - mean_numba
            variance_numba += diff * diff
        variance_numba = variance_numba / scale
        
        print(f"  Numba: window_sum={window_sum:.6f}, mean={mean_numba:.6f}, variance={variance_numba:.6f}")
        print(f"  Window comparison: {np.array([test_data[i-j] for j in range(scale)])}")
        print(f"  Match: {np.allclose([variance_ref], [variance_numba], rtol=1e-10)}")

if __name__ == "__main__":
    debug_variance_implementation()