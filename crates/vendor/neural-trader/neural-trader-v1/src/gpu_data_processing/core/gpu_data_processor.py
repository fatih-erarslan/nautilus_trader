"""
GPU-Accelerated Data Processor using cuDF

This module provides GPU-optimized market data ingestion and processing
with cuDF DataFrames for maximum performance.
"""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import warnings

try:
    import cudf
    import cupy as cp
    import numba.cuda as cuda
    from numba import cuda as nb_cuda
    import pandas as pd
    import numpy as np
    GPU_AVAILABLE = cuda.is_available()
except ImportError as e:
    GPU_AVAILABLE = False
    import pandas as pd
    import numpy as np
    warnings.warn(f"GPU libraries not available: {e}. Using CPU fallback.")

logger = logging.getLogger(__name__)


class GPUDataProcessor:
    """
    GPU-accelerated market data processor using cuDF for maximum performance.
    
    Features:
    - cuDF DataFrame operations for GPU-native data processing
    - Vectorized OHLCV data transformations
    - Memory-efficient batch processing
    - Automatic fallback to CPU when GPU unavailable
    - Real-time data pipeline optimization
    
    Performance Targets:
    - 5,000x+ speedup vs pandas
    - Process 100,000+ rows in <1 second
    - Memory efficiency >70%
    """
    
    def __init__(self, 
                 memory_pool_size: str = "8GB",
                 enable_fallback: bool = True,
                 batch_size: int = 50000):
        """
        Initialize GPU Data Processor.
        
        Args:
            memory_pool_size: GPU memory pool size
            enable_fallback: Enable CPU fallback if GPU unavailable
            batch_size: Batch size for processing large datasets
        """
        self.gpu_available = GPU_AVAILABLE
        self.enable_fallback = enable_fallback
        self.batch_size = batch_size
        self.processing_stats = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "gpu_utilization": 0.0,
            "memory_efficiency": 0.0
        }
        
        if self.gpu_available:
            try:
                # Initialize GPU memory pool
                self._setup_gpu_memory_pool(memory_pool_size)
                logger.info(f"GPU Data Processor initialized with {memory_pool_size} memory pool")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                if not enable_fallback:
                    raise
                self.gpu_available = False
                
        if not self.gpu_available and enable_fallback:
            logger.info("Using CPU fallback mode")
            
    def _setup_gpu_memory_pool(self, pool_size: str):
        """Setup GPU memory pool for efficient memory management."""
        if not self.gpu_available:
            return
            
        try:
            # Parse memory size
            size_gb = float(pool_size.replace("GB", ""))
            pool_size_bytes = int(size_gb * 1024**3)
            
            # Setup cupy memory pool
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=pool_size_bytes)
            
            logger.info(f"GPU memory pool configured: {size_gb}GB")
        except Exception as e:
            logger.error(f"Failed to setup GPU memory pool: {e}")
            
    def process_ohlcv_data(self, 
                          data: Union[pd.DataFrame, cudf.DataFrame],
                          symbols: Optional[List[str]] = None) -> Union[pd.DataFrame, cudf.DataFrame]:
        """
        Process OHLCV market data with GPU acceleration.
        
        Args:
            data: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
            symbols: Optional list of symbols to filter
            
        Returns:
            Processed DataFrame with additional calculated columns
        """
        start_time = time.time()
        
        try:
            # Convert to cuDF if GPU available
            if self.gpu_available and isinstance(data, pd.DataFrame):
                df = cudf.from_pandas(data)
            elif self.gpu_available:
                df = data
            else:
                df = data if isinstance(data, pd.DataFrame) else data.to_pandas()
                
            # Filter symbols if specified
            if symbols and 'Symbol' in df.columns:
                df = df[df['Symbol'].isin(symbols)]
                
            # Calculate additional OHLCV features
            df = self._calculate_ohlcv_features(df)
            
            # Update processing stats
            processing_time = time.time() - start_time
            self._update_stats(len(df), processing_time)
            
            logger.info(f"Processed {len(df)} OHLCV records in {processing_time:.4f}s")
            return df
            
        except Exception as e:
            logger.error(f"OHLCV processing failed: {e}")
            if self.enable_fallback and self.gpu_available:
                # Fallback to CPU
                logger.info("Falling back to CPU processing")
                return self._process_ohlcv_cpu_fallback(data, symbols)
            raise
            
    def _calculate_ohlcv_features(self, df: Union[pd.DataFrame, cudf.DataFrame]) -> Union[pd.DataFrame, cudf.DataFrame]:
        """Calculate additional OHLCV features using vectorized operations."""
        if self.gpu_available and hasattr(df, 'assign'):
            # GPU-accelerated calculations using cuDF
            df = df.assign(
                # Price features
                hl_avg=(df['High'] + df['Low']) / 2,
                ohlc_avg=(df['Open'] + df['High'] + df['Low'] + df['Close']) / 4,
                price_range=df['High'] - df['Low'],
                price_change=df['Close'] - df['Open'],
                price_change_pct=(df['Close'] - df['Open']) / df['Open'] * 100,
                
                # Volume features
                volume_price=df['Volume'] * df['Close'],
                volume_weighted_price=df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3,
                
                # Volatility proxy
                true_range_approx=cp.maximum(
                    df['High'] - df['Low'],
                    cp.maximum(
                        cp.abs(df['High'] - df['Close'].shift(1)),
                        cp.abs(df['Low'] - df['Close'].shift(1))
                    )
                ),
                
                # Gap detection
                gap_up=(df['Open'] > df['Close'].shift(1)) & (df['Open'] - df['Close'].shift(1) > df['Close'].shift(1) * 0.02),
                gap_down=(df['Open'] < df['Close'].shift(1)) & (df['Close'].shift(1) - df['Open'] > df['Close'].shift(1) * 0.02),
            )
        else:
            # CPU fallback calculations
            df = df.copy()
            df['hl_avg'] = (df['High'] + df['Low']) / 2
            df['ohlc_avg'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
            df['price_range'] = df['High'] - df['Low']
            df['price_change'] = df['Close'] - df['Open']
            df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['volume_price'] = df['Volume'] * df['Close']
            df['volume_weighted_price'] = df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3
            
            # True range approximation
            prev_close = df['Close'].shift(1)
            df['true_range_approx'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    np.abs(df['High'] - prev_close),
                    np.abs(df['Low'] - prev_close)
                )
            )
            
            # Gap detection
            df['gap_up'] = (df['Open'] > prev_close) & (df['Open'] - prev_close > prev_close * 0.02)
            df['gap_down'] = (df['Open'] < prev_close) & (prev_close - df['Open'] > prev_close * 0.02)
            
        return df
        
    def _process_ohlcv_cpu_fallback(self, data, symbols):
        """CPU fallback for OHLCV processing."""
        df = data if isinstance(data, pd.DataFrame) else data.to_pandas()
        
        if symbols and 'Symbol' in df.columns:
            df = df[df['Symbol'].isin(symbols)]
            
        return self._calculate_ohlcv_features(df)
        
    def batch_process_market_data(self, 
                                 data_batches: List[Union[pd.DataFrame, cudf.DataFrame]],
                                 parallel: bool = True) -> List[Union[pd.DataFrame, cudf.DataFrame]]:
        """
        Process multiple batches of market data in parallel.
        
        Args:
            data_batches: List of DataFrames to process
            parallel: Enable parallel processing on GPU
            
        Returns:
            List of processed DataFrames
        """
        start_time = time.time()
        results = []
        
        try:
            for i, batch in enumerate(data_batches):
                logger.debug(f"Processing batch {i+1}/{len(data_batches)}")
                processed_batch = self.process_ohlcv_data(batch)
                results.append(processed_batch)
                
                # Memory cleanup for GPU
                if self.gpu_available and i % 10 == 0:
                    cp.get_default_memory_pool().free_all_blocks()
                    
            processing_time = time.time() - start_time
            total_rows = sum(len(batch) for batch in data_batches)
            
            logger.info(f"Batch processed {total_rows} rows in {processing_time:.4f}s")
            logger.info(f"Throughput: {total_rows/processing_time:.0f} rows/second")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
            
    def calculate_rolling_statistics(self, 
                                   df: Union[pd.DataFrame, cudf.DataFrame],
                                   column: str,
                                   windows: List[int] = [5, 10, 20, 50]) -> Union[pd.DataFrame, cudf.DataFrame]:
        """
        Calculate rolling statistics with GPU acceleration.
        
        Args:
            df: Input DataFrame
            column: Column name for calculations
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling statistics
        """
        start_time = time.time()
        
        try:
            result_df = df.copy()
            
            for window in windows:
                if self.gpu_available and hasattr(df, 'rolling'):
                    # GPU-accelerated rolling calculations
                    rolling = df[column].rolling(window=window)
                    result_df[f'{column}_sma_{window}'] = rolling.mean()
                    result_df[f'{column}_std_{window}'] = rolling.std()
                    result_df[f'{column}_min_{window}'] = rolling.min()
                    result_df[f'{column}_max_{window}'] = rolling.max()
                else:
                    # CPU fallback
                    rolling = df[column].rolling(window=window)
                    result_df[f'{column}_sma_{window}'] = rolling.mean()
                    result_df[f'{column}_std_{window}'] = rolling.std()
                    result_df[f'{column}_min_{window}'] = rolling.min()
                    result_df[f'{column}_max_{window}'] = rolling.max()
                    
            processing_time = time.time() - start_time
            logger.debug(f"Rolling statistics calculated in {processing_time:.4f}s")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Rolling statistics calculation failed: {e}")
            raise
            
    def normalize_data(self, 
                      df: Union[pd.DataFrame, cudf.DataFrame],
                      columns: List[str],
                      method: str = "z_score") -> Union[pd.DataFrame, cudf.DataFrame]:
        """
        Normalize data using GPU-accelerated operations.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method ('z_score', 'min_max', 'robust')
            
        Returns:
            DataFrame with normalized columns
        """
        start_time = time.time()
        result_df = df.copy()
        
        try:
            for column in columns:
                if method == "z_score":
                    if self.gpu_available:
                        mean_val = df[column].mean()
                        std_val = df[column].std()
                        result_df[f'{column}_normalized'] = (df[column] - mean_val) / std_val
                    else:
                        mean_val = df[column].mean()
                        std_val = df[column].std()
                        result_df[f'{column}_normalized'] = (df[column] - mean_val) / std_val
                        
                elif method == "min_max":
                    if self.gpu_available:
                        min_val = df[column].min()
                        max_val = df[column].max()
                        result_df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
                    else:
                        min_val = df[column].min()
                        max_val = df[column].max()
                        result_df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
                        
                elif method == "robust":
                    if self.gpu_available:
                        median_val = df[column].median()
                        mad_val = (df[column] - median_val).abs().median()
                        result_df[f'{column}_normalized'] = (df[column] - median_val) / mad_val
                    else:
                        median_val = df[column].median()
                        mad_val = (df[column] - median_val).abs().median()
                        result_df[f'{column}_normalized'] = (df[column] - median_val) / mad_val
                        
            processing_time = time.time() - start_time
            logger.debug(f"Data normalization completed in {processing_time:.4f}s")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            raise
            
    def correlation_matrix(self, 
                          df: Union[pd.DataFrame, cudf.DataFrame],
                          columns: Optional[List[str]] = None) -> Union[pd.DataFrame, cudf.DataFrame]:
        """
        Calculate correlation matrix with GPU acceleration.
        
        Args:
            df: Input DataFrame
            columns: Columns to include in correlation matrix
            
        Returns:
            Correlation matrix
        """
        start_time = time.time()
        
        try:
            if columns:
                data = df[columns]
            else:
                # Select only numeric columns
                if self.gpu_available:
                    numeric_cols = df.select_dtypes(include=[cp.number]).columns
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                data = df[numeric_cols]
                
            if self.gpu_available and hasattr(data, 'corr'):
                corr_matrix = data.corr()
            else:
                corr_matrix = data.corr()
                
            processing_time = time.time() - start_time
            logger.debug(f"Correlation matrix calculated in {processing_time:.4f}s")
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            raise
            
    def detect_outliers(self, 
                       df: Union[pd.DataFrame, cudf.DataFrame],
                       column: str,
                       method: str = "iqr",
                       threshold: float = 1.5) -> Union[pd.DataFrame, cudf.DataFrame]:
        """
        Detect outliers using GPU-accelerated statistical methods.
        
        Args:
            df: Input DataFrame
            column: Column to analyze for outliers
            method: Detection method ('iqr', 'z_score', 'modified_z_score')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        start_time = time.time()
        result_df = df.copy()
        
        try:
            if method == "iqr":
                if self.gpu_available:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    result_df[f'{column}_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
                else:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    result_df[f'{column}_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
                    
            elif method == "z_score":
                mean_val = df[column].mean()
                std_val = df[column].std()
                if self.gpu_available:
                    z_scores = cp.abs((df[column] - mean_val) / std_val)
                    result_df[f'{column}_outlier'] = z_scores > threshold
                else:
                    z_scores = np.abs((df[column] - mean_val) / std_val)
                    result_df[f'{column}_outlier'] = z_scores > threshold
                    
            processing_time = time.time() - start_time
            logger.debug(f"Outlier detection completed in {processing_time:.4f}s")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            raise
            
    def _update_stats(self, rows_processed: int, processing_time: float):
        """Update processing statistics."""
        self.processing_stats["total_processed"] += rows_processed
        
        # Update average processing time (exponential moving average)
        alpha = 0.1
        self.processing_stats["avg_processing_time"] = (
            alpha * processing_time + 
            (1 - alpha) * self.processing_stats["avg_processing_time"]
        )
        
        # Calculate throughput
        throughput = rows_processed / processing_time if processing_time > 0 else 0
        
        # Update GPU utilization (if available)
        if self.gpu_available:
            try:
                # This is a simplified metric - in practice would use nvidia-ml-py
                self.processing_stats["gpu_utilization"] = min(throughput / 100000, 1.0)
                
                # Memory efficiency approximation
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                if total_bytes > 0:
                    self.processing_stats["memory_efficiency"] = used_bytes / total_bytes
            except Exception:
                pass
                
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.processing_stats.copy()
        stats.update({
            "gpu_available": self.gpu_available,
            "batch_size": self.batch_size,
            "enable_fallback": self.enable_fallback
        })
        
        if self.gpu_available:
            try:
                stats["gpu_memory_info"] = {
                    "used_bytes": cp.get_default_memory_pool().used_bytes(),
                    "total_bytes": cp.get_default_memory_pool().total_bytes(),
                    "free_bytes": cp.get_default_memory_pool().total_bytes() - cp.get_default_memory_pool().used_bytes()
                }
            except Exception:
                pass
                
        return stats
        
    def optimize_memory_usage(self):
        """Optimize GPU memory usage."""
        if self.gpu_available:
            try:
                # Clear memory pool
                cp.get_default_memory_pool().free_all_blocks()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info("GPU memory optimized")
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")
                
    def benchmark_performance(self, 
                            data_size: int = 100000,
                            iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance.
        
        Args:
            data_size: Size of test dataset
            iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        logger.info(f"Starting performance benchmark with {data_size} rows")
        
        # Generate test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, data_size),
            'High': np.random.uniform(150, 250, data_size),
            'Low': np.random.uniform(50, 150, data_size),
            'Close': np.random.uniform(100, 200, data_size),
            'Volume': np.random.randint(1000, 1000000, data_size)
        })
        
        # GPU benchmark
        gpu_times = []
        if self.gpu_available:
            for i in range(iterations):
                start_time = time.time()
                gpu_df = cudf.from_pandas(test_data)
                self.process_ohlcv_data(gpu_df)
                gpu_times.append(time.time() - start_time)
                
                # Memory cleanup
                del gpu_df
                cp.get_default_memory_pool().free_all_blocks()
                
        # CPU benchmark
        cpu_times = []
        original_gpu_state = self.gpu_available
        self.gpu_available = False  # Force CPU mode
        
        for i in range(iterations):
            start_time = time.time()
            self.process_ohlcv_data(test_data.copy())
            cpu_times.append(time.time() - start_time)
            
        self.gpu_available = original_gpu_state
        
        # Calculate metrics
        avg_gpu_time = np.mean(gpu_times) if gpu_times else float('inf')
        avg_cpu_time = np.mean(cpu_times)
        speedup = avg_cpu_time / avg_gpu_time if gpu_times else 0
        
        throughput_gpu = data_size / avg_gpu_time if gpu_times else 0
        throughput_cpu = data_size / avg_cpu_time
        
        results = {
            "data_size": data_size,
            "iterations": iterations,
            "avg_gpu_time": avg_gpu_time,
            "avg_cpu_time": avg_cpu_time,
            "speedup_factor": speedup,
            "throughput_gpu": throughput_gpu,
            "throughput_cpu": throughput_cpu,
            "target_speedup_achieved": speedup >= 1000,  # 1000x minimum target
            "target_throughput_achieved": throughput_gpu >= 100000  # 100k rows/sec target
        }
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  GPU Time: {avg_gpu_time:.4f}s")
        logger.info(f"  CPU Time: {avg_cpu_time:.4f}s") 
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  GPU Throughput: {throughput_gpu:.0f} rows/sec")
        
        return results
        
    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'gpu_available') and self.gpu_available:
            try:
                self.optimize_memory_usage()
            except Exception:
                pass