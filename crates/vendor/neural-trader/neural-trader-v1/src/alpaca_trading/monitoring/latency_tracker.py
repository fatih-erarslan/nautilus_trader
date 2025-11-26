"""
Latency tracking for high-frequency trading operations.
Provides microsecond precision timing and performance analysis.
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import numpy as np
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Represents a single latency measurement."""
    operation: str
    start_time: float
    end_time: float
    duration_us: float  # microseconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_us / 1000
    
    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_us / 1_000_000


class LatencyTracker:
    """
    High-precision latency tracking with histogram generation and percentile analysis.
    """
    
    def __init__(self, window_size: int = 10000):
        """
        Initialize latency tracker.
        
        Args:
            window_size: Number of measurements to keep in memory
        """
        self.window_size = window_size
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.active_measurements: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
    def start_measurement(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata for the measurement
            
        Returns:
            Measurement ID
        """
        measurement_id = f"{operation}_{time.time_ns()}"
        self.active_measurements[measurement_id] = time.perf_counter_ns()
        return measurement_id
    
    async def async_start_measurement(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Async version of start_measurement."""
        async with self._lock:
            return self.start_measurement(operation, metadata)
    
    def end_measurement(self, measurement_id: str, metadata: Optional[Dict[str, Any]] = None) -> LatencyMeasurement:
        """
        End timing an operation.
        
        Args:
            measurement_id: ID returned by start_measurement
            metadata: Optional metadata to add to the measurement
            
        Returns:
            LatencyMeasurement object
        """
        if measurement_id not in self.active_measurements:
            raise ValueError(f"Unknown measurement ID: {measurement_id}")
        
        start_time_ns = self.active_measurements.pop(measurement_id)
        end_time_ns = time.perf_counter_ns()
        duration_ns = end_time_ns - start_time_ns
        duration_us = duration_ns / 1000
        
        operation = measurement_id.split('_')[0]
        
        measurement = LatencyMeasurement(
            operation=operation,
            start_time=start_time_ns / 1_000_000_000,  # Convert to seconds
            end_time=end_time_ns / 1_000_000_000,
            duration_us=duration_us,
            metadata=metadata or {}
        )
        
        self.measurements[operation].append(measurement)
        return measurement
    
    async def async_end_measurement(self, measurement_id: str, metadata: Optional[Dict[str, Any]] = None) -> LatencyMeasurement:
        """Async version of end_measurement."""
        async with self._lock:
            return self.end_measurement(measurement_id, metadata)
    
    def measure(self, operation: str):
        """
        Context manager for measuring operation latency.
        
        Usage:
            with tracker.measure('order_send'):
                # code to measure
        """
        class MeasurementContext:
            def __init__(self, tracker, operation):
                self.tracker = tracker
                self.operation = operation
                self.measurement_id = None
                self.measurement = None
                
            def __enter__(self):
                self.measurement_id = self.tracker.start_measurement(self.operation)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.measurement = self.tracker.end_measurement(self.measurement_id)
                
        return MeasurementContext(self, operation)
    
    def async_measure(self, operation: str):
        """
        Async context manager for measuring operation latency.
        
        Usage:
            async with tracker.async_measure('order_send'):
                # async code to measure
        """
        class AsyncMeasurementContext:
            def __init__(self, tracker, operation):
                self.tracker = tracker
                self.operation = operation
                self.measurement_id = None
                self.measurement = None
                
            async def __aenter__(self):
                self.measurement_id = await self.tracker.async_start_measurement(self.operation)
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.measurement = await self.tracker.async_end_measurement(self.measurement_id)
                
        return AsyncMeasurementContext(self, operation)
    
    def get_percentiles(self, operation: str, percentiles: List[float] = None) -> Dict[str, float]:
        """
        Calculate percentiles for an operation.
        
        Args:
            operation: Operation name
            percentiles: List of percentiles to calculate (default: [50, 95, 99])
            
        Returns:
            Dictionary of percentile values in microseconds
        """
        if percentiles is None:
            percentiles = [50, 95, 99]
            
        measurements = self.measurements.get(operation, [])
        if not measurements:
            return {f"p{p}": 0.0 for p in percentiles}
        
        durations = [m.duration_us for m in measurements]
        results = {}
        
        for p in percentiles:
            value = np.percentile(durations, p)
            results[f"p{p}"] = value
            
        return results
    
    def get_statistics(self, operation: str) -> Dict[str, float]:
        """
        Get comprehensive statistics for an operation.
        
        Returns:
            Dictionary with min, max, mean, median, std, p50, p95, p99
        """
        measurements = self.measurements.get(operation, [])
        if not measurements:
            return {
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        durations = [m.duration_us for m in measurements]
        
        return {
            'count': len(durations),
            'min': min(durations),
            'max': max(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'std': statistics.stdev(durations) if len(durations) > 1 else 0.0,
            **self.get_percentiles(operation)
        }
    
    def generate_histogram(self, operation: str, bins: int = 20) -> Dict[str, Any]:
        """
        Generate histogram data for an operation.
        
        Args:
            operation: Operation name
            bins: Number of histogram bins
            
        Returns:
            Dictionary with bin edges and counts
        """
        measurements = self.measurements.get(operation, [])
        if not measurements:
            return {'edges': [], 'counts': [], 'total': 0}
        
        durations = [m.duration_us for m in measurements]
        counts, edges = np.histogram(durations, bins=bins)
        
        return {
            'edges': edges.tolist(),
            'counts': counts.tolist(),
            'total': len(durations)
        }
    
    def get_stage_breakdown(self, prefix: str) -> Dict[str, Dict[str, float]]:
        """
        Get latency breakdown by stages (operations with common prefix).
        
        Args:
            prefix: Common prefix for related operations (e.g., 'order_')
            
        Returns:
            Dictionary of statistics for each stage
        """
        breakdown = {}
        
        for operation in self.measurements:
            if operation.startswith(prefix):
                breakdown[operation] = self.get_statistics(operation)
                
        return breakdown
    
    def get_all_operations(self) -> List[str]:
        """Get list of all tracked operations."""
        return list(self.measurements.keys())
    
    def clear_operation(self, operation: str):
        """Clear measurements for a specific operation."""
        if operation in self.measurements:
            self.measurements[operation].clear()
    
    def clear_all(self):
        """Clear all measurements."""
        self.measurements.clear()
        self.active_measurements.clear()
    
    def export_measurements(self, operation: str = None) -> List[Dict[str, Any]]:
        """
        Export measurements as a list of dictionaries.
        
        Args:
            operation: Specific operation to export (None for all)
            
        Returns:
            List of measurement dictionaries
        """
        operations = [operation] if operation else self.get_all_operations()
        
        exported = []
        for op in operations:
            for measurement in self.measurements.get(op, []):
                exported.append({
                    'operation': measurement.operation,
                    'start_time': measurement.start_time,
                    'end_time': measurement.end_time,
                    'duration_us': measurement.duration_us,
                    'duration_ms': measurement.duration_ms,
                    'metadata': measurement.metadata
                })
                
        return exported


class StageLatencyTracker:
    """
    Track latency across multiple stages of a process.
    """
    
    def __init__(self, tracker: LatencyTracker):
        """
        Initialize stage tracker.
        
        Args:
            tracker: Main latency tracker instance
        """
        self.tracker = tracker
        self.stages: Dict[str, List[str]] = {}
        self.active_stages: Dict[str, str] = {}
    
    def start_stage(self, process_id: str, stage: str) -> str:
        """Start timing a stage in a process."""
        measurement_id = self.tracker.start_measurement(f"{process_id}_{stage}")
        
        if process_id not in self.stages:
            self.stages[process_id] = []
        
        self.stages[process_id].append(stage)
        self.active_stages[f"{process_id}_{stage}"] = measurement_id
        
        return measurement_id
    
    def end_stage(self, process_id: str, stage: str, metadata: Optional[Dict[str, Any]] = None):
        """End timing a stage in a process."""
        key = f"{process_id}_{stage}"
        if key not in self.active_stages:
            raise ValueError(f"Stage not started: {key}")
        
        measurement_id = self.active_stages.pop(key)
        return self.tracker.end_measurement(measurement_id, metadata)
    
    def get_process_breakdown(self, process_id: str) -> Dict[str, Dict[str, float]]:
        """Get latency breakdown for all stages in a process."""
        if process_id not in self.stages:
            return {}
        
        breakdown = {}
        for stage in self.stages[process_id]:
            operation = f"{process_id}_{stage}"
            breakdown[stage] = self.tracker.get_statistics(operation)
            
        return breakdown
    
    def get_total_latency(self, process_id: str) -> float:
        """Get total latency across all stages of a process."""
        breakdown = self.get_process_breakdown(process_id)
        return sum(stats['mean'] for stats in breakdown.values())