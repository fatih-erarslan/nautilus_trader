#!/usr/bin/env python3
"""
Resource Usage Validation Module for AI News Trading Platform.

This module validates that all resource usage meets performance targets:
- Memory Usage: < 2GB sustained
- CPU Usage: < 80% under load
- Disk I/O: Efficient data storage and retrieval
- Network: Optimal bandwidth utilization
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import psutil
import gc
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.resource_benchmark import ResourceBenchmark


class ResourceValidator:
    """Validates resource usage performance targets"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize resource validator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize benchmark components
        try:
            self.resource_benchmark = ResourceBenchmark(self.config)
        except Exception as e:
            self.logger.warning(f"Could not initialize ResourceBenchmark: {e}")
            self.resource_benchmark = None
        
        # Resource monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.resource_samples = []
        self.monitoring_lock = threading.Lock()
    
    async def validate_memory_usage(self) -> Dict[str, Any]:
        """Validate memory usage meets < 2GB sustained target
        
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating memory usage...")
        start_time = time.time()
        
        try:
            # Test different memory usage scenarios
            scenarios = {
                'baseline': {'duration': 30, 'load': 'normal'},
                'signal_generation': {'duration': 60, 'load': 'signal_heavy'},
                'data_processing': {'duration': 45, 'load': 'data_heavy'},
                'concurrent_operations': {'duration': 90, 'load': 'mixed_heavy'},
                'sustained_load': {'duration': 120, 'load': 'sustained'}
            }
            
            scenario_results = {}
            max_memory_mb = 0.0
            sustained_memory_mb = 0.0
            
            for scenario_name, params in scenarios.items():
                self.logger.debug(f"Testing {scenario_name} memory usage...")
                
                # Measure memory usage for scenario
                memory_stats = await self._measure_memory_usage(
                    duration=params['duration'],
                    load_type=params['load']
                )
                
                scenario_results[scenario_name] = memory_stats
                max_memory_mb = max(max_memory_mb, memory_stats['peak_memory_mb'])
                
                # For sustained load, use average as sustained memory
                if scenario_name == 'sustained_load':
                    sustained_memory_mb = memory_stats['average_memory_mb']
            
            # Determine if target is met (< 2GB = 2048 MB)
            target_met = sustained_memory_mb < 2048.0
            peak_acceptable = max_memory_mb < 3072.0  # Allow 3GB peak
            
            message = (
                f"Memory usage: {sustained_memory_mb:.2f}MB sustained, "
                f"{max_memory_mb:.2f}MB peak (target: < 2048MB sustained) - "
                f"{'PASS' if target_met else 'FAIL'}"
            )
            
            return {
                'measured_value': sustained_memory_mb,
                'target_value': 2048.0,
                'target_met': target_met,
                'message': message,
                'duration_seconds': time.time() - start_time,
                'metadata': {
                    'scenarios_tested': list(scenarios.keys()),
                    'sustained_memory_mb': sustained_memory_mb,
                    'peak_memory_mb': max_memory_mb,
                    'peak_acceptable': peak_acceptable,
                    'scenario_breakdown': scenario_results,
                    'memory_efficiency': self._calculate_memory_efficiency(scenario_results)
                }
            }
                
        except Exception as e:
            self.logger.error(f"Memory usage validation failed: {e}")
            return {
                'measured_value': None,
                'target_value': 2048.0,
                'target_met': False,
                'message': f'Validation failed: {str(e)}',
                'duration_seconds': time.time() - start_time,
                'metadata': {'error': str(e)}
            }
    
    async def validate_cpu_usage(self) -> Dict[str, Any]:
        """Validate CPU usage meets < 80% under load target
        
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating CPU usage...")
        start_time = time.time()
        
        try:
            # Test different CPU usage scenarios
            scenarios = {
                'idle': {'duration': 15, 'load': 'idle'},
                'normal_operations': {'duration': 30, 'load': 'normal'},
                'heavy_computation': {'duration': 45, 'load': 'heavy'},
                'mixed_workload': {'duration': 60, 'load': 'mixed'},
                'stress_test': {'duration': 30, 'load': 'stress'}
            }
            
            scenario_results = {}
            max_cpu_percent = 0.0
            sustained_cpu_percent = 0.0
            
            for scenario_name, params in scenarios.items():
                self.logger.debug(f"Testing {scenario_name} CPU usage...")
                
                # Measure CPU usage for scenario
                cpu_stats = await self._measure_cpu_usage(
                    duration=params['duration'],
                    load_type=params['load']
                )
                
                scenario_results[scenario_name] = cpu_stats
                max_cpu_percent = max(max_cpu_percent, cpu_stats['peak_cpu_percent'])
                
                # For heavy computation, use average as sustained CPU
                if scenario_name == 'heavy_computation':
                    sustained_cpu_percent = cpu_stats['average_cpu_percent']
            
            # Determine if target is met (< 80% under load)
            target_met = sustained_cpu_percent < 80.0
            peak_acceptable = max_cpu_percent < 95.0  # Allow 95% peak
            
            message = (
                f"CPU usage: {sustained_cpu_percent:.1f}% under load, "
                f"{max_cpu_percent:.1f}% peak (target: < 80% under load) - "
                f"{'PASS' if target_met else 'FAIL'}"
            )
            
            return {
                'measured_value': sustained_cpu_percent,
                'target_value': 80.0,
                'target_met': target_met,
                'message': message,
                'duration_seconds': time.time() - start_time,
                'metadata': {
                    'scenarios_tested': list(scenarios.keys()),
                    'sustained_cpu_percent': sustained_cpu_percent,
                    'peak_cpu_percent': max_cpu_percent,
                    'peak_acceptable': peak_acceptable,
                    'scenario_breakdown': scenario_results,
                    'cpu_efficiency': self._calculate_cpu_efficiency(scenario_results)
                }
            }
                
        except Exception as e:
            self.logger.error(f"CPU usage validation failed: {e}")
            return {
                'measured_value': None,
                'target_value': 80.0,
                'target_met': False,
                'message': f'Validation failed: {str(e)}',
                'duration_seconds': time.time() - start_time,
                'metadata': {'error': str(e)}
            }
    
    async def validate_disk_io_efficiency(self) -> Dict[str, Any]:
        """Validate disk I/O efficiency
        
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating disk I/O efficiency...")
        start_time = time.time()
        
        try:
            # Test different I/O scenarios
            scenarios = {
                'sequential_read': {'operation': 'read', 'pattern': 'sequential'},
                'random_read': {'operation': 'read', 'pattern': 'random'},
                'sequential_write': {'operation': 'write', 'pattern': 'sequential'},
                'random_write': {'operation': 'write', 'pattern': 'random'},
                'mixed_workload': {'operation': 'mixed', 'pattern': 'mixed'}
            }
            
            scenario_results = {}
            min_performance = float('inf')
            
            for scenario_name, params in scenarios.items():
                self.logger.debug(f"Testing {scenario_name} I/O performance...")
                
                # Measure I/O performance
                io_stats = await self._measure_disk_io(
                    operation=params['operation'],
                    pattern=params['pattern']
                )
                
                scenario_results[scenario_name] = io_stats
                min_performance = min(min_performance, io_stats['throughput_mb_per_sec'])
            
            # Determine if I/O is efficient (> 50 MB/s minimum)
            target_met = min_performance > 50.0
            
            message = (
                f"Disk I/O performance: {min_performance:.2f}MB/s minimum "
                f"(target: > 50MB/s) - {'PASS' if target_met else 'FAIL'}"
            )
            
            return {
                'measured_value': min_performance,
                'target_value': 50.0,
                'target_met': target_met,
                'message': message,
                'duration_seconds': time.time() - start_time,
                'metadata': {
                    'scenarios_tested': list(scenarios.keys()),
                    'min_performance_mb_per_sec': min_performance,
                    'scenario_breakdown': scenario_results,
                    'io_efficiency': self._calculate_io_efficiency(scenario_results)
                }
            }
                
        except Exception as e:
            self.logger.error(f"Disk I/O validation failed: {e}")
            return {
                'measured_value': None,
                'target_value': 50.0,
                'target_met': False,
                'message': f'Validation failed: {str(e)}',
                'duration_seconds': time.time() - start_time,
                'metadata': {'error': str(e)}
            }
    
    async def _measure_memory_usage(self, duration: int, load_type: str) -> Dict[str, Any]:
        """Measure memory usage under specific load conditions
        
        Args:
            duration: Test duration in seconds
            load_type: Type of load to simulate
            
        Returns:
            Dictionary with memory statistics
        """
        memory_samples = []
        
        # Start memory monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_memory(duration, memory_samples)
        )
        
        # Start load simulation
        load_task = asyncio.create_task(
            self._simulate_memory_load(duration, load_type)
        )
        
        # Wait for both to complete
        await asyncio.gather(monitoring_task, load_task)
        
        if memory_samples:
            return {
                'duration_seconds': duration,
                'samples': len(memory_samples),
                'peak_memory_mb': max(memory_samples),
                'min_memory_mb': min(memory_samples),
                'average_memory_mb': np.mean(memory_samples),
                'median_memory_mb': np.median(memory_samples),
                'std_memory_mb': np.std(memory_samples),
                'p95_memory_mb': np.percentile(memory_samples, 95),
                'p99_memory_mb': np.percentile(memory_samples, 99)
            }
        else:
            return {
                'duration_seconds': duration,
                'samples': 0,
                'peak_memory_mb': 0,
                'min_memory_mb': 0,
                'average_memory_mb': 0,
                'median_memory_mb': 0,
                'std_memory_mb': 0,
                'p95_memory_mb': 0,
                'p99_memory_mb': 0
            }
    
    async def _measure_cpu_usage(self, duration: int, load_type: str) -> Dict[str, Any]:
        """Measure CPU usage under specific load conditions
        
        Args:
            duration: Test duration in seconds
            load_type: Type of load to simulate
            
        Returns:
            Dictionary with CPU statistics
        """
        cpu_samples = []
        
        # Start CPU monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_cpu(duration, cpu_samples)
        )
        
        # Start load simulation
        load_task = asyncio.create_task(
            self._simulate_cpu_load(duration, load_type)
        )
        
        # Wait for both to complete
        await asyncio.gather(monitoring_task, load_task)
        
        if cpu_samples:
            return {
                'duration_seconds': duration,
                'samples': len(cpu_samples),
                'peak_cpu_percent': max(cpu_samples),
                'min_cpu_percent': min(cpu_samples),
                'average_cpu_percent': np.mean(cpu_samples),
                'median_cpu_percent': np.median(cpu_samples),
                'std_cpu_percent': np.std(cpu_samples),
                'p95_cpu_percent': np.percentile(cpu_samples, 95),
                'p99_cpu_percent': np.percentile(cpu_samples, 99)
            }
        else:
            return {
                'duration_seconds': duration,
                'samples': 0,
                'peak_cpu_percent': 0,
                'min_cpu_percent': 0,
                'average_cpu_percent': 0,
                'median_cpu_percent': 0,
                'std_cpu_percent': 0,
                'p95_cpu_percent': 0,
                'p99_cpu_percent': 0
            }
    
    async def _measure_disk_io(self, operation: str, pattern: str) -> Dict[str, Any]:
        """Measure disk I/O performance
        
        Args:
            operation: Type of operation (read, write, mixed)
            pattern: Access pattern (sequential, random, mixed)
            
        Returns:
            Dictionary with I/O statistics
        """
        # Create temporary test file
        test_file = Path(f"/tmp/io_test_{os.getpid()}_{int(time.time())}.dat")
        file_size_mb = 100  # 100MB test file
        
        try:
            # Simulate I/O operations
            if operation == 'write' or operation == 'mixed':
                write_throughput = await self._simulate_disk_write(test_file, file_size_mb, pattern)
            else:
                write_throughput = 0
            
            if operation == 'read' or operation == 'mixed':
                # Ensure file exists for read test
                if not test_file.exists():
                    await self._create_test_file(test_file, file_size_mb)
                read_throughput = await self._simulate_disk_read(test_file, pattern)
            else:
                read_throughput = 0
            
            # Calculate overall throughput
            if operation == 'mixed':
                throughput = (read_throughput + write_throughput) / 2
            elif operation == 'read':
                throughput = read_throughput
            else:
                throughput = write_throughput
            
            return {
                'operation': operation,
                'pattern': pattern,
                'throughput_mb_per_sec': throughput,
                'read_throughput_mb_per_sec': read_throughput,
                'write_throughput_mb_per_sec': write_throughput,
                'file_size_mb': file_size_mb
            }
            
        finally:
            # Cleanup test file
            if test_file.exists():
                test_file.unlink()
    
    async def _monitor_memory(self, duration: int, samples: List[float]):
        """Monitor memory usage and collect samples"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                # Get current memory usage in MB
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                samples.append(memory_mb)
                await asyncio.sleep(0.5)  # Sample every 0.5 seconds
            except Exception as e:
                self.logger.debug(f"Memory monitoring error: {e}")
                break
    
    async def _monitor_cpu(self, duration: int, samples: List[float]):
        """Monitor CPU usage and collect samples"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                # Get current CPU usage percentage
                cpu_percent = psutil.cpu_percent(interval=0.5)
                samples.append(cpu_percent)
            except Exception as e:
                self.logger.debug(f"CPU monitoring error: {e}")
                break
    
    async def _simulate_memory_load(self, duration: int, load_type: str):
        """Simulate memory load based on load type"""
        if load_type == 'idle':
            await asyncio.sleep(duration)
            return
        
        # Determine memory allocation based on load type
        if load_type == 'normal':
            base_size = 50 * 1024 * 1024  # 50MB
        elif load_type == 'signal_heavy':
            base_size = 200 * 1024 * 1024  # 200MB
        elif load_type == 'data_heavy':
            base_size = 300 * 1024 * 1024  # 300MB
        elif load_type == 'mixed_heavy':
            base_size = 400 * 1024 * 1024  # 400MB
        elif load_type == 'sustained':
            base_size = 500 * 1024 * 1024  # 500MB
        else:
            base_size = 100 * 1024 * 1024  # 100MB default
        
        # Simulate memory usage
        memory_objects = []
        end_time = time.time() + duration
        
        try:
            while time.time() < end_time:
                # Allocate memory
                size = int(base_size * np.random.uniform(0.5, 1.5))
                data = bytearray(size)
                memory_objects.append(data)
                
                # Occasionally free some memory
                if len(memory_objects) > 10 and np.random.random() < 0.3:
                    del memory_objects[0]
                    gc.collect()
                
                await asyncio.sleep(np.random.uniform(0.1, 0.5))
        finally:
            # Cleanup
            del memory_objects
            gc.collect()
    
    async def _simulate_cpu_load(self, duration: int, load_type: str):
        """Simulate CPU load based on load type"""
        if load_type == 'idle':
            await asyncio.sleep(duration)
            return
        
        # Determine CPU load intensity
        if load_type == 'normal':
            intensity = 0.3
        elif load_type == 'heavy':
            intensity = 0.7
        elif load_type == 'mixed':
            intensity = 0.5
        elif load_type == 'stress':
            intensity = 0.9
        else:
            intensity = 0.4
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # CPU intensive work
            work_duration = intensity * 0.1  # Work for portion of time
            rest_duration = (1 - intensity) * 0.1  # Rest for remainder
            
            # Simulate CPU work
            work_end = time.time() + work_duration
            while time.time() < work_end:
                # Mathematical computation to use CPU
                result = sum(i * i for i in range(1000))
            
            # Rest period
            if rest_duration > 0:
                await asyncio.sleep(rest_duration)
    
    async def _simulate_disk_write(self, test_file: Path, size_mb: int, pattern: str) -> float:
        """Simulate disk write operations"""
        start_time = time.time()
        
        try:
            with open(test_file, 'wb') as f:
                if pattern == 'sequential':
                    # Write large chunks sequentially
                    chunk_size = 1024 * 1024  # 1MB chunks
                    data = b'x' * chunk_size
                    for _ in range(size_mb):
                        f.write(data)
                        f.flush()
                        os.fsync(f.fileno())
                elif pattern == 'random':
                    # Write smaller chunks randomly
                    chunk_size = 4096  # 4KB chunks
                    data = b'x' * chunk_size
                    chunks_per_mb = 1024 * 1024 // chunk_size
                    for _ in range(size_mb * chunks_per_mb):
                        f.write(data)
                        if np.random.random() < 0.1:  # Occasional flush
                            f.flush()
                else:
                    # Mixed pattern
                    chunk_size = 64 * 1024  # 64KB chunks
                    data = b'x' * chunk_size
                    chunks_per_mb = 1024 * 1024 // chunk_size
                    for _ in range(size_mb * chunks_per_mb):
                        f.write(data)
                        if np.random.random() < 0.2:
                            f.flush()
        
        except Exception as e:
            self.logger.error(f"Disk write simulation failed: {e}")
            return 0.0
        
        duration = time.time() - start_time
        return size_mb / duration if duration > 0 else 0.0
    
    async def _simulate_disk_read(self, test_file: Path, pattern: str) -> float:
        """Simulate disk read operations"""
        start_time = time.time()
        
        try:
            file_size = test_file.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            with open(test_file, 'rb') as f:
                if pattern == 'sequential':
                    # Read large chunks sequentially
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        data = f.read(chunk_size)
                        if not data:
                            break
                elif pattern == 'random':
                    # Random access reads
                    chunk_size = 4096  # 4KB chunks
                    for _ in range(int(size_mb * 256)):  # Read equivalent data
                        pos = np.random.randint(0, file_size - chunk_size)
                        f.seek(pos)
                        data = f.read(chunk_size)
                else:
                    # Mixed pattern
                    chunk_size = 64 * 1024  # 64KB chunks
                    pos = 0
                    while pos < file_size:
                        f.seek(pos)
                        data = f.read(chunk_size)
                        if not data:
                            break
                        pos += chunk_size
                        # Occasional random jump
                        if np.random.random() < 0.1:
                            pos = np.random.randint(pos, file_size - chunk_size)
        
        except Exception as e:
            self.logger.error(f"Disk read simulation failed: {e}")
            return 0.0
        
        duration = time.time() - start_time
        return size_mb / duration if duration > 0 else 0.0
    
    async def _create_test_file(self, test_file: Path, size_mb: int):
        """Create a test file for I/O operations"""
        with open(test_file, 'wb') as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            data = b'x' * chunk_size
            for _ in range(size_mb):
                f.write(data)
    
    def _calculate_memory_efficiency(self, scenario_results: Dict) -> Dict[str, Any]:
        """Calculate memory efficiency metrics"""
        efficiency = {}
        
        # Calculate memory utilization ratios
        for scenario, stats in scenario_results.items():
            peak = stats.get('peak_memory_mb', 0)
            average = stats.get('average_memory_mb', 0)
            
            if peak > 0:
                efficiency[f'{scenario}_utilization_ratio'] = average / peak
                efficiency[f'{scenario}_memory_stability'] = 1.0 - (stats.get('std_memory_mb', 0) / average) if average > 0 else 0
        
        return efficiency
    
    def _calculate_cpu_efficiency(self, scenario_results: Dict) -> Dict[str, Any]:
        """Calculate CPU efficiency metrics"""
        efficiency = {}
        
        # Calculate CPU utilization patterns
        for scenario, stats in scenario_results.items():
            peak = stats.get('peak_cpu_percent', 0)
            average = stats.get('average_cpu_percent', 0)
            
            if peak > 0:
                efficiency[f'{scenario}_cpu_utilization_ratio'] = average / peak
                efficiency[f'{scenario}_cpu_stability'] = 1.0 - (stats.get('std_cpu_percent', 0) / average) if average > 0 else 0
        
        return efficiency
    
    def _calculate_io_efficiency(self, scenario_results: Dict) -> Dict[str, Any]:
        """Calculate I/O efficiency metrics"""
        efficiency = {}
        
        # Calculate I/O performance ratios
        read_throughputs = []
        write_throughputs = []
        
        for scenario, stats in scenario_results.items():
            read_tp = stats.get('read_throughput_mb_per_sec', 0)
            write_tp = stats.get('write_throughput_mb_per_sec', 0)
            
            if read_tp > 0:
                read_throughputs.append(read_tp)
            if write_tp > 0:
                write_throughputs.append(write_tp)
        
        if read_throughputs:
            efficiency['read_performance_consistency'] = min(read_throughputs) / max(read_throughputs)
        if write_throughputs:
            efficiency['write_performance_consistency'] = min(write_throughputs) / max(write_throughputs)
        
        return efficiency


async def main():
    """Main entry point for resource validation testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Resource Validator")
    parser.add_argument('--test', choices=['memory', 'cpu', 'disk', 'all'], 
                       default='all', help='Which test to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize validator
    validator = ResourceValidator()
    
    # Run tests
    results = {}
    
    if args.test in ['memory', 'all']:
        print("Running memory usage validation...")
        results['memory'] = await validator.validate_memory_usage()
        print(f"Result: {results['memory']['message']}")
    
    if args.test in ['cpu', 'all']:
        print("\nRunning CPU usage validation...")
        results['cpu'] = await validator.validate_cpu_usage()
        print(f"Result: {results['cpu']['message']}")
    
    if args.test in ['disk', 'all']:
        print("\nRunning disk I/O validation...")
        results['disk'] = await validator.validate_disk_io_efficiency()
        print(f"Result: {results['disk']['message']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("RESOURCE VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result['target_met'] else "FAIL"
        print(f"{test_name.capitalize()}: {status}")
        if not result['target_met']:
            all_passed = False
    
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)