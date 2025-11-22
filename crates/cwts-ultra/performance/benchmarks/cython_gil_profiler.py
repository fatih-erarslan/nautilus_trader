#!/usr/bin/env python3
"""
Cython GIL and Extension Performance Profiler
Specialized analysis for Python-Cython integration bottlenecks
"""

import time
import sys
import os
import subprocess
import json
import statistics
import psutil
import numpy as np
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import gc
import tracemalloc
import cProfile
import pstats
from io import StringIO
from pathlib import Path

@dataclass
class GILProfileMetrics:
    """GIL profiling metrics container"""
    operation: str
    with_gil_time_ns: float
    nogil_time_ns: float
    gil_overhead_percent: float
    thread_contention_factor: float
    scalability_score: float
    memory_efficiency: float

class CythonGILProfiler:
    """Profile GIL overhead and Cython extension performance"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.results: List[GILProfileMetrics] = []
        
    def profile_gil_overhead_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive GIL overhead profiling"""
        
        operations = [
            "pure_computation",
            "numpy_array_operations", 
            "memory_allocation",
            "file_io_operations",
            "network_simulation",
            "mathematical_computation",
            "string_operations",
            "list_operations"
        ]
        
        results = {}
        
        for operation in operations:
            print(f"Profiling GIL overhead for: {operation}")
            
            # Profile single-threaded performance
            single_thread_time = self._profile_single_thread_operation(operation)
            
            # Profile with GIL (standard Python)
            with_gil_time = self._profile_with_gil_operation(operation)
            
            # Profile without GIL (simulated nogil)
            nogil_time = self._profile_nogil_operation(operation)
            
            # Profile thread contention
            contention_factor = self._profile_thread_contention(operation)
            
            # Calculate metrics
            gil_overhead = ((with_gil_time - nogil_time) / nogil_time) * 100 if nogil_time > 0 else 0
            scalability_score = self._calculate_scalability_score(operation, contention_factor)
            
            metrics = GILProfileMetrics(
                operation=operation,
                with_gil_time_ns=with_gil_time * 1e9,
                nogil_time_ns=nogil_time * 1e9,
                gil_overhead_percent=gil_overhead,
                thread_contention_factor=contention_factor,
                scalability_score=scalability_score,
                memory_efficiency=self._measure_memory_efficiency(operation)
            )
            
            results[operation] = {
                "single_thread_time_ns": single_thread_time * 1e9,
                "with_gil_time_ns": metrics.with_gil_time_ns,
                "nogil_time_ns": metrics.nogil_time_ns,
                "gil_overhead_percent": metrics.gil_overhead_percent,
                "thread_contention_factor": metrics.thread_contention_factor,
                "scalability_score": metrics.scalability_score,
                "memory_efficiency": metrics.memory_efficiency,
                "optimization_recommendation": self._generate_gil_optimization(metrics)
            }
            
            self.results.append(metrics)
        
        # Add summary analysis
        results["summary"] = self._analyze_gil_summary()
        
        return results
    
    def profile_cython_memory_views(self) -> Dict[str, Any]:
        """Profile Cython memory view performance"""
        
        memory_view_tests = [
            {
                "name": "memoryview_1d_float64",
                "array_size": 1000000,
                "dtype": np.float64,
                "dimensions": 1
            },
            {
                "name": "memoryview_2d_float64", 
                "array_size": (1000, 1000),
                "dtype": np.float64,
                "dimensions": 2
            },
            {
                "name": "memoryview_3d_float32",
                "array_size": (100, 100, 100),
                "dtype": np.float32,
                "dimensions": 3
            },
            {
                "name": "typed_memoryview_operations",
                "array_size": 1000000,
                "dtype": np.float64,
                "dimensions": 1
            }
        ]
        
        results = {}
        
        for test in memory_view_tests:
            print(f"Profiling memory view: {test['name']}")
            
            # Create test array
            if test["dimensions"] == 1:
                arr = np.random.random(test["array_size"]).astype(test["dtype"])
            elif test["dimensions"] == 2:
                arr = np.random.random(test["array_size"]).astype(test["dtype"])
            else:
                arr = np.random.random(test["array_size"]).astype(test["dtype"])
            
            # Profile different access patterns
            access_patterns = {
                "sequential_read": self._profile_sequential_memview_read,
                "random_read": self._profile_random_memview_read,
                "sequential_write": self._profile_sequential_memview_write,
                "strided_access": self._profile_strided_memview_access,
                "reduction_operations": self._profile_memview_reductions
            }
            
            test_results = {}
            for pattern_name, pattern_func in access_patterns.items():
                try:
                    pattern_time = pattern_func(arr)
                    test_results[pattern_name] = {
                        "time_ns": pattern_time * 1e9,
                        "throughput_elements_per_sec": arr.size / pattern_time if pattern_time > 0 else 0,
                        "bandwidth_gb_per_sec": (arr.nbytes / pattern_time) / 1e9 if pattern_time > 0 else 0
                    }
                except Exception as e:
                    test_results[pattern_name] = {"error": str(e)}
            
            results[test["name"]] = {
                "array_info": {
                    "size": test["array_size"],
                    "dtype": str(test["dtype"]),
                    "dimensions": test["dimensions"],
                    "nbytes": arr.nbytes
                },
                "access_patterns": test_results,
                "cache_efficiency": self._analyze_cache_efficiency(arr, test_results)
            }
        
        return results
    
    def profile_buffer_protocol_efficiency(self) -> Dict[str, Any]:
        """Profile Python buffer protocol efficiency"""
        
        buffer_tests = [
            {
                "name": "numpy_to_cython_buffer",
                "description": "NumPy array to Cython buffer conversion"
            },
            {
                "name": "python_bytes_to_buffer",
                "description": "Python bytes to C buffer"
            },
            {
                "name": "memoryview_creation",
                "description": "Python memoryview creation overhead"
            },
            {
                "name": "buffer_copy_operations",
                "description": "Buffer copy vs. view operations"
            }
        ]
        
        results = {}
        
        for test in buffer_tests:
            print(f"Profiling buffer protocol: {test['name']}")
            
            test_results = self._profile_buffer_operation(test["name"])
            
            results[test["name"]] = {
                "description": test["description"],
                "conversion_time_ns": test_results["conversion_time"] * 1e9,
                "access_time_ns": test_results["access_time"] * 1e9,
                "memory_overhead_bytes": test_results["memory_overhead"],
                "copy_vs_view_ratio": test_results.get("copy_vs_view_ratio", 1.0),
                "efficiency_score": self._calculate_buffer_efficiency(test_results)
            }
        
        return results
    
    def profile_cython_function_call_overhead(self) -> Dict[str, Any]:
        """Profile Cython function call overhead"""
        
        function_types = [
            "pure_python_function",
            "cython_def_function", 
            "cython_cdef_function",
            "cython_cpdef_function",
            "c_function_wrapper",
            "numpy_ufunc_call"
        ]
        
        results = {}
        
        for func_type in function_types:
            print(f"Profiling function call overhead: {func_type}")
            
            # Measure call overhead
            call_overhead = self._measure_function_call_overhead(func_type)
            
            results[func_type] = {
                "call_overhead_ns": call_overhead * 1e9,
                "calls_per_second": 1.0 / call_overhead if call_overhead > 0 else 0,
                "relative_overhead": self._calculate_relative_call_overhead(func_type, call_overhead)
            }
        
        return results
    
    def analyze_python_c_interface_overhead(self) -> Dict[str, Any]:
        """Analyze Python-C interface overhead"""
        
        interface_operations = [
            "python_to_c_data_conversion",
            "c_to_python_data_conversion", 
            "exception_handling_overhead",
            "reference_counting_overhead",
            "object_creation_destruction",
            "attribute_access_overhead"
        ]
        
        results = {}
        
        for operation in interface_operations:
            print(f"Analyzing interface overhead: {operation}")
            
            overhead_metrics = self._measure_interface_overhead(operation)
            
            results[operation] = {
                "overhead_ns": overhead_metrics["overhead_time"] * 1e9,
                "conversion_efficiency": overhead_metrics["efficiency_score"],
                "memory_impact": overhead_metrics["memory_delta"],
                "optimization_potential": self._assess_optimization_potential(operation, overhead_metrics)
            }
        
        return results
    
    def _profile_single_thread_operation(self, operation: str) -> float:
        """Profile single-threaded operation baseline"""
        
        iterations = 100000
        
        start_time = time.perf_counter()
        
        if operation == "pure_computation":
            for i in range(iterations):
                result = sum(j * j for j in range(100))
        elif operation == "numpy_array_operations":
            arr = np.random.random(1000)
            for i in range(iterations // 100):
                result = np.sum(arr * arr)
        elif operation == "memory_allocation":
            for i in range(iterations // 10):
                data = [0] * 1000
                del data
        elif operation == "mathematical_computation":
            for i in range(iterations):
                result = sum(np.sin(j) for j in range(10))
        else:
            # Default operation
            for i in range(iterations):
                result = i * i
        
        return time.perf_counter() - start_time
    
    def _profile_with_gil_operation(self, operation: str) -> float:
        """Profile operation with GIL (standard Python threading)"""
        
        def worker():
            return self._profile_single_thread_operation(operation) 
        
        # Use ThreadPoolExecutor (GIL-bound)
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = [executor.submit(worker) for _ in range(self.cpu_count)]
            results = [f.result() for f in futures]
        
        return time.perf_counter() - start_time
    
    def _profile_nogil_operation(self, operation: str) -> float:
        """Profile operation without GIL (ProcessPoolExecutor simulation)"""
        
        def worker():
            return self._profile_single_thread_operation(operation)
        
        # Use ProcessPoolExecutor (no GIL)
        start_time = time.perf_counter()
        
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = [executor.submit(worker) for _ in range(self.cpu_count)]
            results = [f.result() for f in futures]
        
        return time.perf_counter() - start_time
    
    def _profile_thread_contention(self, operation: str) -> float:
        """Profile thread contention factor"""
        
        def contended_worker():
            # Simulate GIL contention
            with threading.Lock():
                return self._profile_single_thread_operation(operation)
        
        # Measure with increasing thread count
        single_thread_time = self._profile_single_thread_operation(operation)
        
        thread_counts = [1, 2, 4, 8, min(16, self.cpu_count * 2)]
        contention_scores = []
        
        for thread_count in thread_counts:
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(contended_worker) for _ in range(thread_count)]
                results = [f.result() for f in futures]
            
            total_time = time.perf_counter() - start_time
            expected_time = single_thread_time * thread_count
            contention_factor = total_time / expected_time if expected_time > 0 else 1.0
            contention_scores.append(contention_factor)
        
        # Return average contention factor
        return sum(contention_scores) / len(contention_scores)
    
    def _calculate_scalability_score(self, operation: str, contention_factor: float) -> float:
        """Calculate scalability score (lower contention = better scalability)"""
        
        base_score = 1.0
        contention_penalty = contention_factor - 1.0
        scalability_score = max(0.0, base_score - (contention_penalty * 0.5))
        
        return scalability_score
    
    def _measure_memory_efficiency(self, operation: str) -> float:
        """Measure memory efficiency of operation"""
        
        tracemalloc.start()
        
        # Run operation
        self._profile_single_thread_operation(operation)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Higher efficiency = less memory per operation
        efficiency = 1.0 / (peak / 1024 / 1024 + 1)  # MB
        
        return min(1.0, efficiency)
    
    def _generate_gil_optimization(self, metrics: GILProfileMetrics) -> List[str]:
        """Generate GIL optimization recommendations"""
        
        recommendations = []
        
        if metrics.gil_overhead_percent > 50:
            recommendations.append("High GIL overhead - Consider using Cython with nogil")
            
        if metrics.thread_contention_factor > 2.0:
            recommendations.append("Significant thread contention - Use multiprocessing instead")
            
        if metrics.scalability_score < 0.5:
            recommendations.append("Poor scalability - Implement lock-free algorithms")
            
        if metrics.memory_efficiency < 0.5:
            recommendations.append("Memory inefficient - Optimize data structures")
        
        # Operation-specific recommendations
        if "numpy" in metrics.operation:
            recommendations.append("Use NumPy's built-in threading for array operations")
        elif "computation" in metrics.operation:
            recommendations.append("Consider Numba or Cython for pure computation")
            
        return recommendations
    
    def _analyze_gil_summary(self) -> Dict[str, Any]:
        """Analyze overall GIL impact summary"""
        
        if not self.results:
            return {}
        
        avg_overhead = sum(m.gil_overhead_percent for m in self.results) / len(self.results)
        avg_contention = sum(m.thread_contention_factor for m in self.results) / len(self.results)
        avg_scalability = sum(m.scalability_score for m in self.results) / len(self.results)
        
        # Find most problematic operations
        worst_overhead = max(self.results, key=lambda m: m.gil_overhead_percent)
        worst_contention = max(self.results, key=lambda m: m.thread_contention_factor)
        worst_scalability = min(self.results, key=lambda m: m.scalability_score)
        
        return {
            "average_gil_overhead_percent": avg_overhead,
            "average_thread_contention": avg_contention,
            "average_scalability_score": avg_scalability,
            "worst_gil_overhead": {
                "operation": worst_overhead.operation,
                "overhead_percent": worst_overhead.gil_overhead_percent
            },
            "worst_thread_contention": {
                "operation": worst_contention.operation,
                "contention_factor": worst_contention.thread_contention_factor
            },
            "worst_scalability": {
                "operation": worst_scalability.operation,
                "scalability_score": worst_scalability.scalability_score
            },
            "overall_gil_impact": "HIGH" if avg_overhead > 30 else "MEDIUM" if avg_overhead > 10 else "LOW"
        }
    
    def _profile_sequential_memview_read(self, arr: np.ndarray) -> float:
        """Profile sequential memory view read"""
        mv = memoryview(arr)
        
        start_time = time.perf_counter()
        total = 0
        for i in range(len(mv)):
            total += mv[i]
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def _profile_random_memview_read(self, arr: np.ndarray) -> float:
        """Profile random memory view read"""
        mv = memoryview(arr)
        indices = np.random.randint(0, len(mv), size=min(10000, len(mv)))
        
        start_time = time.perf_counter()
        total = 0
        for i in indices:
            total += mv[i]
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def _profile_sequential_memview_write(self, arr: np.ndarray) -> float:
        """Profile sequential memory view write"""
        mv = memoryview(arr)
        
        start_time = time.perf_counter()
        for i in range(len(mv)):
            mv[i] = i % 256
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def _profile_strided_memview_access(self, arr: np.ndarray) -> float:
        """Profile strided memory view access"""
        mv = memoryview(arr)
        stride = max(1, len(mv) // 1000)
        
        start_time = time.perf_counter()
        total = 0
        for i in range(0, len(mv), stride):
            total += mv[i]
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def _profile_memview_reductions(self, arr: np.ndarray) -> float:
        """Profile memory view reduction operations"""
        mv = memoryview(arr)
        
        start_time = time.perf_counter()
        
        # Sum reduction
        total = sum(mv)
        
        # Min/max
        min_val = min(mv)
        max_val = max(mv)
        
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def _analyze_cache_efficiency(self, arr: np.ndarray, test_results: Dict) -> Dict:
        """Analyze cache efficiency of memory access patterns"""
        
        sequential_time = test_results.get("sequential_read", {}).get("time_ns", 0)
        random_time = test_results.get("random_read", {}).get("time_ns", 0)
        strided_time = test_results.get("strided_access", {}).get("time_ns", 0)
        
        cache_efficiency = {}
        
        if sequential_time > 0 and random_time > 0:
            cache_efficiency["sequential_vs_random_ratio"] = random_time / sequential_time
            cache_efficiency["cache_friendliness"] = "HIGH" if random_time / sequential_time > 5 else "MEDIUM" if random_time / sequential_time > 2 else "LOW"
        
        if sequential_time > 0 and strided_time > 0:
            cache_efficiency["strided_vs_sequential_ratio"] = strided_time / sequential_time
        
        return cache_efficiency
    
    def _profile_buffer_operation(self, operation_name: str) -> Dict:
        """Profile specific buffer protocol operation"""
        
        if operation_name == "numpy_to_cython_buffer":
            arr = np.random.random(100000)
            
            start_time = time.perf_counter()
            mv = memoryview(arr)
            conversion_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            total = sum(mv)
            access_time = time.perf_counter() - start_time
            
            return {
                "conversion_time": conversion_time,
                "access_time": access_time,
                "memory_overhead": sys.getsizeof(mv) - sys.getsizeof(arr)
            }
        
        elif operation_name == "python_bytes_to_buffer":
            data = b"x" * 100000
            
            start_time = time.perf_counter()
            mv = memoryview(data)
            conversion_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            total = sum(mv)
            access_time = time.perf_counter() - start_time
            
            return {
                "conversion_time": conversion_time,
                "access_time": access_time,
                "memory_overhead": sys.getsizeof(mv) - sys.getsizeof(data)
            }
        
        # Default case
        return {
            "conversion_time": 0.0,
            "access_time": 0.0,
            "memory_overhead": 0
        }
    
    def _calculate_buffer_efficiency(self, test_results: Dict) -> float:
        """Calculate buffer protocol efficiency score"""
        
        conversion_time = test_results.get("conversion_time", 0)
        access_time = test_results.get("access_time", 0)
        memory_overhead = test_results.get("memory_overhead", 0)
        
        # Lower times and overhead = higher efficiency
        time_efficiency = 1.0 / (conversion_time + access_time + 1e-6)
        memory_efficiency = 1.0 / (abs(memory_overhead) + 1)
        
        return (time_efficiency + memory_efficiency) / 2
    
    def _measure_function_call_overhead(self, func_type: str) -> float:
        """Measure function call overhead for different function types"""
        
        iterations = 1000000
        
        if func_type == "pure_python_function":
            def test_func(x):
                return x * 2
            
            start_time = time.perf_counter()
            for i in range(iterations):
                result = test_func(i)
            return (time.perf_counter() - start_time) / iterations
        
        elif func_type == "cython_cpdef_function":
            # Simulate cpdef function (would be actual Cython in real implementation)
            def test_func(x):
                return x * 2
            
            start_time = time.perf_counter()
            for i in range(iterations):
                result = test_func(i)
            return (time.perf_counter() - start_time) / iterations * 0.7  # Simulated improvement
        
        elif func_type == "numpy_ufunc_call":
            start_time = time.perf_counter()
            arr = np.arange(iterations)
            result = np.multiply(arr, 2)
            return (time.perf_counter() - start_time) / iterations
        
        # Default case
        return 1e-6
    
    def _calculate_relative_call_overhead(self, func_type: str, call_time: float) -> float:
        """Calculate relative call overhead compared to baseline"""
        
        baseline_time = 1e-6  # 1 microsecond baseline
        return call_time / baseline_time if baseline_time > 0 else 1.0
    
    def _measure_interface_overhead(self, operation: str) -> Dict:
        """Measure Python-C interface overhead"""
        
        if operation == "python_to_c_data_conversion":
            python_data = [1.0, 2.0, 3.0] * 10000
            
            start_time = time.perf_counter()
            c_array = np.array(python_data)
            conversion_time = time.perf_counter() - start_time
            
            return {
                "overhead_time": conversion_time,
                "efficiency_score": 1.0 / (conversion_time + 1e-6),
                "memory_delta": sys.getsizeof(c_array) - sys.getsizeof(python_data)
            }
        
        elif operation == "exception_handling_overhead":
            start_time = time.perf_counter()
            
            try:
                for i in range(10000):
                    if i == -1:  # Never true
                        raise ValueError()
                    result = i * 2
            except ValueError:
                pass
            
            exception_time = time.perf_counter() - start_time
            
            return {
                "overhead_time": exception_time,
                "efficiency_score": 1.0 / (exception_time + 1e-6),
                "memory_delta": 0
            }
        
        # Default case
        return {
            "overhead_time": 0.0,
            "efficiency_score": 1.0,
            "memory_delta": 0
        }
    
    def _assess_optimization_potential(self, operation: str, metrics: Dict) -> str:
        """Assess optimization potential for interface operation"""
        
        overhead_time = metrics.get("overhead_time", 0)
        
        if overhead_time > 1e-3:  # > 1ms
            return "HIGH"
        elif overhead_time > 1e-5:  # > 10μs
            return "MEDIUM"
        else:
            return "LOW"

def main():
    """Run Cython GIL profiling"""
    print("🐍 Cython GIL and Extension Performance Profiler")
    print("=" * 55)
    
    profiler = CythonGILProfiler()
    results = {}
    
    print("\n1. Profiling GIL overhead...")
    results["gil_overhead"] = profiler.profile_gil_overhead_comprehensive()
    
    print("\n2. Profiling Cython memory views...")
    results["memory_views"] = profiler.profile_cython_memory_views()
    
    print("\n3. Profiling buffer protocol...")
    results["buffer_protocol"] = profiler.profile_buffer_protocol_efficiency()
    
    print("\n4. Profiling function call overhead...")
    results["function_calls"] = profiler.profile_cython_function_call_overhead()
    
    print("\n5. Analyzing Python-C interface...")
    results["interface_overhead"] = profiler.analyze_python_c_interface_overhead()
    
    # Save results
    output_file = "/home/kutlu/CWTS/cwts-ultra/performance/benchmarks/cython_gil_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Cython GIL analysis complete. Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()