"""
Real-time Neural Data Processor

This module provides real-time data processing capabilities for neural
forecasting models, enabling low-latency inference and streaming data handling.

Key Features:
- Real-time data streaming
- Low-latency preprocessing
- Sliding window management
- Online learning support
- Performance monitoring
"""

import asyncio
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union, AsyncIterator, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for real-time processing."""
    # Window management
    window_size: int = 480  # Size of sliding window
    update_frequency: float = 1.0  # Update frequency in Hz
    
    # Processing settings
    batch_size: int = 1  # Batch size for real-time processing
    max_latency_ms: float = 100.0  # Maximum allowed latency
    
    # Buffer settings
    buffer_size: int = 1000  # Size of data buffer
    overflow_strategy: str = "drop_oldest"  # "drop_oldest", "drop_newest", "block"
    
    # Quality control
    enable_quality_checks: bool = True
    quality_check_interval: int = 10  # Check every N samples
    
    # Performance optimization
    use_threading: bool = True
    max_workers: int = 2
    memory_limit_mb: int = 100


@dataclass
class ProcessingResult:
    """Result of real-time processing."""
    processed_data: np.ndarray
    timestamp: float
    latency_ms: float
    sequence_id: int
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingMetrics:
    """Metrics for streaming performance."""
    total_processed: int = 0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_hz: float = 0.0
    buffer_utilization: float = 0.0
    quality_score: float = 1.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


class RealtimeNeuralProcessor:
    """
    Real-time neural data processor for streaming inference.
    
    This class handles real-time data streams, applies neural preprocessing,
    and maintains sliding windows for continuous inference.
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sliding window buffer
        self.data_buffer = deque(maxlen=config.buffer_size)
        self.window_buffer = deque(maxlen=config.window_size)
        
        # Processing components (will be injected)
        self.preprocessor = None
        self.normalizer = None
        self.feature_engineer = None
        
        # Streaming state
        self.is_streaming = False
        self.sequence_counter = 0
        
        # Performance tracking
        self.metrics = StreamingMetrics()
        self.latency_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        self.start_time = None
        
        # Threading
        self.executor = None
        if config.use_threading:
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Quality monitoring
        self.quality_history = deque(maxlen=50)
        self.last_quality_check = 0
        
        self.logger.info("RealtimeNeuralProcessor initialized")
    
    def set_preprocessing_components(
        self,
        preprocessor=None,
        normalizer=None,
        feature_engineer=None
    ):
        """Set preprocessing components for the pipeline."""
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        self.feature_engineer = feature_engineer
        
        self.logger.info("Preprocessing components configured")
    
    def add_data_callback(self, callback: Callable[[ProcessingResult], None]):
        """Add callback for processed data."""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors."""
        self.error_callbacks.append(callback)
    
    async def start_streaming(self):
        """Start real-time streaming processing."""
        try:
            if self.is_streaming:
                self.logger.warning("Streaming already active")
                return
            
            self.is_streaming = True
            self.start_time = time.time()
            self.sequence_counter = 0
            
            self.logger.info("Starting real-time streaming")
            
            # Start processing loop
            await self._streaming_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting streaming: {e}")
            await self.stop_streaming()
            for callback in self.error_callbacks:
                callback(e)
    
    async def stop_streaming(self):
        """Stop real-time streaming processing."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Update final metrics
        if self.start_time:
            self.metrics.uptime_seconds = time.time() - self.start_time
        
        self.logger.info("Stopped real-time streaming")
    
    async def process_data_point(self, data_point: Union[Dict[str, float], pd.Series]) -> Optional[ProcessingResult]:
        """
        Process a single data point in real-time.
        
        Args:
            data_point: New data point to process
        
        Returns:
            ProcessingResult if window is ready, None otherwise
        """
        start_time = time.perf_counter()
        
        try:
            # Add to buffer
            self._add_to_buffer(data_point)
            
            # Check if we have enough data for processing
            if len(self.window_buffer) < self.config.window_size:
                return None
            
            # Process current window
            result = await self._process_window()
            
            # Calculate metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self.latency_history.append(processing_time)
            self.processing_times.append(processing_time)
            
            # Update metrics
            self._update_metrics(processing_time)
            
            # Quality check
            if (self.sequence_counter % self.config.quality_check_interval == 0 and 
                self.config.enable_quality_checks):
                await self._quality_check()
            
            # Call callbacks
            if result:
                for callback in self.data_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing data point: {e}")
            for callback in self.error_callbacks:
                callback(e)
            return None
    
    async def process_batch(self, data_batch: List[Union[Dict[str, float], pd.Series]]) -> List[ProcessingResult]:
        """
        Process a batch of data points.
        
        Args:
            data_batch: Batch of data points
        
        Returns:
            List of processing results
        """
        results = []
        
        for data_point in data_batch:
            result = await self.process_data_point(data_point)
            if result:
                results.append(result)
        
        return results
    
    async def _streaming_loop(self):
        """Main streaming processing loop."""
        update_interval = 1.0 / self.config.update_frequency
        
        while self.is_streaming:
            try:
                loop_start = time.time()
                
                # Process any pending data in buffer
                if self.data_buffer:
                    # Process in batches
                    batch_size = min(self.config.batch_size, len(self.data_buffer))
                    batch = []
                    
                    for _ in range(batch_size):
                        if self.data_buffer:
                            batch.append(self.data_buffer.popleft())
                    
                    if batch:
                        await self.process_batch(batch)
                
                # Calculate sleep time to maintain frequency
                loop_time = time.time() - loop_start
                sleep_time = max(0, update_interval - loop_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in streaming loop: {e}")
                for callback in self.error_callbacks:
                    callback(e)
    
    def _add_to_buffer(self, data_point: Union[Dict[str, float], pd.Series]):
        """Add data point to buffer."""
        try:
            # Convert to standard format
            if isinstance(data_point, pd.Series):
                data_dict = data_point.to_dict()
            elif isinstance(data_point, dict):
                data_dict = data_point.copy()
            else:
                # Convert other types
                data_dict = {'value': float(data_point)}
            
            # Add timestamp if not present
            if 'timestamp' not in data_dict:
                data_dict['timestamp'] = time.time()
            
            # Add to window buffer
            self.window_buffer.append(data_dict)
            
            # Handle buffer overflow
            if len(self.data_buffer) >= self.config.buffer_size:
                if self.config.overflow_strategy == "drop_oldest":
                    self.data_buffer.popleft()
                elif self.config.overflow_strategy == "drop_newest":
                    return  # Don't add new data
                # "block" strategy would wait, but we'll just drop oldest for now
            
            self.data_buffer.append(data_dict)
            
        except Exception as e:
            self.logger.error(f"Error adding to buffer: {e}")
    
    async def _process_window(self) -> Optional[ProcessingResult]:
        """Process current window of data."""
        try:
            if len(self.window_buffer) < self.config.window_size:
                return None
            
            # Convert window to numpy array
            window_data = list(self.window_buffer)
            
            # Extract features (assume first non-timestamp field is primary)
            feature_keys = [k for k in window_data[0].keys() if k != 'timestamp']
            if not feature_keys:
                return None
            
            # Create feature matrix
            feature_matrix = np.zeros((len(window_data), len(feature_keys)))
            timestamps = []
            
            for i, point in enumerate(window_data):
                timestamps.append(point.get('timestamp', time.time()))
                for j, key in enumerate(feature_keys):
                    feature_matrix[i, j] = point.get(key, 0.0)
            
            # Apply preprocessing if available
            processed_data = feature_matrix
            
            if self.feature_engineer:
                # Apply feature engineering (simplified for real-time)
                try:
                    # Create a minimal DataFrame for feature engineering
                    df = pd.DataFrame(feature_matrix, columns=feature_keys)
                    df.index = pd.to_datetime(timestamps, unit='s')
                    
                    # Apply feature engineering (would need async version)
                    # For now, just use raw data
                    pass
                except Exception as e:
                    self.logger.warning(f"Feature engineering failed: {e}")
            
            if self.normalizer:
                # Apply normalization (would need to use fitted scaler)
                try:
                    # Use simple z-score normalization for real-time
                    mean = np.mean(processed_data, axis=0)
                    std = np.std(processed_data, axis=0)
                    std[std == 0] = 1  # Avoid division by zero
                    processed_data = (processed_data - mean) / std
                except Exception as e:
                    self.logger.warning(f"Normalization failed: {e}")
            
            # Create result
            self.sequence_counter += 1
            
            result = ProcessingResult(
                processed_data=processed_data,
                timestamp=time.time(),
                latency_ms=0.0,  # Will be updated by caller
                sequence_id=self.sequence_counter,
                metadata={
                    'feature_keys': feature_keys,
                    'window_size': len(window_data),
                    'data_shape': processed_data.shape
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing window: {e}")
            return None
    
    def _update_metrics(self, processing_time_ms: float):
        """Update streaming metrics."""
        self.metrics.total_processed += 1
        
        # Update latency metrics
        if self.latency_history:
            self.metrics.average_latency_ms = np.mean(self.latency_history)
            self.metrics.max_latency_ms = np.max(self.latency_history)
        
        # Update throughput
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.metrics.throughput_hz = self.metrics.total_processed / elapsed_time
        
        # Update buffer utilization
        self.metrics.buffer_utilization = len(self.data_buffer) / self.config.buffer_size
        
        # Update uptime
        if self.start_time:
            self.metrics.uptime_seconds = time.time() - self.start_time
        
        # Check for performance issues
        if processing_time_ms > self.config.max_latency_ms:
            self.logger.warning(f"High latency detected: {processing_time_ms:.2f}ms")
    
    async def _quality_check(self):
        """Perform quality check on processed data."""
        try:
            if not self.window_buffer:
                return
            
            # Simple quality metrics
            quality_score = 1.0
            
            # Check for missing data
            total_points = len(self.window_buffer)
            valid_points = sum(1 for point in self.window_buffer if point and 'timestamp' in point)
            
            if total_points > 0:
                completeness = valid_points / total_points
                quality_score *= completeness
            
            # Check for reasonable data ranges
            try:
                feature_keys = [k for k in self.window_buffer[0].keys() if k != 'timestamp']
                for key in feature_keys:
                    values = [point.get(key, 0) for point in self.window_buffer if key in point]
                    if values:
                        # Check for extreme values
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        
                        if std_val > 0:
                            # Check for outliers (simple z-score)
                            outliers = sum(1 for v in values if abs(v - mean_val) > 3 * std_val)
                            outlier_ratio = outliers / len(values)
                            quality_score *= (1 - outlier_ratio)
            
            except Exception:
                quality_score *= 0.8  # Penalize if quality check fails
            
            self.quality_history.append(quality_score)
            self.metrics.quality_score = np.mean(self.quality_history) if self.quality_history else 1.0
            
            # Log quality issues
            if quality_score < 0.8:
                self.logger.warning(f"Data quality issue detected: score={quality_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
    
    def get_current_window(self) -> Optional[np.ndarray]:
        """Get current window data as numpy array."""
        try:
            if len(self.window_buffer) < self.config.window_size:
                return None
            
            window_data = list(self.window_buffer)
            feature_keys = [k for k in window_data[0].keys() if k != 'timestamp']
            
            if not feature_keys:
                return None
            
            feature_matrix = np.zeros((len(window_data), len(feature_keys)))
            
            for i, point in enumerate(window_data):
                for j, key in enumerate(feature_keys):
                    feature_matrix[i, j] = point.get(key, 0.0)
            
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"Error getting current window: {e}")
            return None
    
    def reset_buffers(self):
        """Reset all buffers and counters."""
        self.data_buffer.clear()
        self.window_buffer.clear()
        self.sequence_counter = 0
        self.latency_history.clear()
        self.processing_times.clear()
        self.quality_history.clear()
        
        # Reset metrics
        self.metrics = StreamingMetrics()
        
        self.logger.info("Buffers reset")
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            'data_buffer_size': len(self.data_buffer),
            'data_buffer_capacity': self.config.buffer_size,
            'window_buffer_size': len(self.window_buffer),
            'window_buffer_capacity': self.config.window_size,
            'buffer_utilization': len(self.data_buffer) / self.config.buffer_size,
            'window_ready': len(self.window_buffer) >= self.config.window_size,
            'sequence_counter': self.sequence_counter
        }
    
    def get_performance_metrics(self) -> StreamingMetrics:
        """Get current performance metrics."""
        # Update error rate
        if self.metrics.total_processed > 0:
            # Simple error rate calculation (would need proper error tracking)
            self.metrics.error_rate = 0.0  # Placeholder
        
        return self.metrics
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics information."""
        return {
            'streaming_active': self.is_streaming,
            'buffer_status': self.get_buffer_status(),
            'performance_metrics': {
                'total_processed': self.metrics.total_processed,
                'average_latency_ms': self.metrics.average_latency_ms,
                'max_latency_ms': self.metrics.max_latency_ms,
                'throughput_hz': self.metrics.throughput_hz,
                'quality_score': self.metrics.quality_score,
                'uptime_seconds': self.metrics.uptime_seconds
            },
            'recent_latencies': list(self.latency_history)[-10:],  # Last 10 latencies
            'config': {
                'window_size': self.config.window_size,
                'update_frequency': self.config.update_frequency,
                'max_latency_ms': self.config.max_latency_ms,
                'buffer_size': self.config.buffer_size
            },
            'components': {
                'preprocessor_loaded': self.preprocessor is not None,
                'normalizer_loaded': self.normalizer is not None,
                'feature_engineer_loaded': self.feature_engineer is not None
            }
        }
    
    async def simulate_data_stream(
        self,
        duration_seconds: float = 60.0,
        data_frequency_hz: float = 10.0
    ):
        """
        Simulate a data stream for testing purposes.
        
        Args:
            duration_seconds: How long to simulate
            data_frequency_hz: Frequency of simulated data
        """
        self.logger.info(f"Starting data stream simulation for {duration_seconds}s at {data_frequency_hz}Hz")
        
        await self.start_streaming()
        
        start_time = time.time()
        update_interval = 1.0 / data_frequency_hz
        sample_count = 0
        
        try:
            while (time.time() - start_time) < duration_seconds and self.is_streaming:
                # Generate synthetic data point
                timestamp = time.time()
                
                # Simple sine wave with noise for testing
                base_value = np.sin(2 * np.pi * sample_count / 100) * 10
                noise = np.random.normal(0, 1)
                
                data_point = {
                    'timestamp': timestamp,
                    'price': base_value + noise + 100,  # Price around 100
                    'volume': np.abs(noise * 1000 + 5000),  # Volume
                    'feature1': np.random.normal(0, 1),
                    'feature2': np.random.normal(1, 0.5)
                }
                
                await self.process_data_point(data_point)
                
                sample_count += 1
                
                # Maintain frequency
                await asyncio.sleep(update_interval)
                
        except Exception as e:
            self.logger.error(f"Error in data simulation: {e}")
        
        finally:
            await self.stop_streaming()
            self.logger.info("Data stream simulation completed")
    
    def __del__(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=False)