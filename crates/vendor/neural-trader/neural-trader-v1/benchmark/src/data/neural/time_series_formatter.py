"""
Time Series Formatter for Neural Networks

This module provides specialized time series data formatting for neural forecasting
models, particularly optimized for NHITS (Neural Hierarchical Interpolation for Time Series).

Key Features:
- Sliding window generation for supervised learning
- Multi-scale windowing for hierarchical models
- Efficient batch processing
- Memory-optimized operations
- Support for multivariate time series
- Seasonal decomposition awareness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series formatting."""
    # Window configuration
    input_size: int = 480  # Lookback window (40 hours for 5-min data)
    forecast_horizon: int = 96  # Forecast horizon (8 hours for 5-min data)
    step_size: int = 1  # Step size for sliding window
    
    # Frequency and seasonality
    frequency: str = "5min"  # Data frequency
    seasonality: List[int] = field(default_factory=lambda: [288, 2016])  # Daily, weekly
    
    # Processing options
    overlap_ratio: float = 0.8  # Overlap between windows
    min_window_completeness: float = 0.9  # Minimum data completeness per window
    pad_sequences: bool = True  # Pad sequences to fixed length
    
    # Multi-scale options for NHITS
    enable_multi_scale: bool = True
    scale_factors: List[int] = field(default_factory=lambda: [1, 4, 8])  # Different time scales
    
    # Memory optimization
    use_float32: bool = True  # Use float32 instead of float64
    chunk_processing: bool = True
    chunk_size: int = 10000
    
    def __post_init__(self):
        # Validate configuration
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if not 0 < self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be between 0 and 1")


@dataclass
class FormattingResult:
    """Result of time series formatting."""
    success: bool
    data: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_names: Optional[List[str]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TimeSeriesFormatter:
    """
    Time series formatter optimized for neural forecasting models.
    
    This class handles the conversion of raw time series data into properly
    formatted input-output pairs suitable for neural network training and
    inference, with special optimizations for NHITS and similar models.
    """
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Calculate derived parameters
        self.step_size = max(1, int(config.input_size * (1 - config.overlap_ratio)))
        
        # Validation
        self._validate_config()
        
        self.logger.info(f"TimeSeriesFormatter initialized with {config.input_size} input, "
                        f"{config.forecast_horizon} forecast horizon")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.input_size < self.config.forecast_horizon:
            warnings.warn("Input size is smaller than forecast horizon")
        
        if self.step_size <= 0:
            raise ValueError("Calculated step size must be positive")
    
    async def format_for_neural_network(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        return_dates: bool = True
    ) -> FormattingResult:
        """
        Format time series data for neural network training/inference.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Column name to forecast
            feature_columns: Additional feature columns to include
            return_dates: Whether to return corresponding dates
        
        Returns:
            FormattingResult containing formatted data
        """
        try:
            self.logger.info(f"Formatting time series: {len(data)} samples")
            
            # Validate inputs
            validation_result = self._validate_input(data, target_column, feature_columns)
            if not validation_result['valid']:
                return FormattingResult(
                    success=False,
                    errors=validation_result['errors']
                )
            
            # Prepare data
            prepared_data = self._prepare_data(data, target_column, feature_columns)
            
            # Generate windows
            if self.config.chunk_processing:
                windows_result = await self._generate_windows_chunked(prepared_data, return_dates)
            else:
                windows_result = self._generate_windows(prepared_data, return_dates)
            
            if not windows_result['success']:
                return FormattingResult(
                    success=False,
                    errors=windows_result['errors']
                )
            
            formatted_data = windows_result['data']
            
            # Add multi-scale windows for NHITS
            if self.config.enable_multi_scale:
                multi_scale_data = self._generate_multi_scale_windows(formatted_data)
                formatted_data.update(multi_scale_data)
            
            # Add metadata
            metadata = {
                'n_windows': len(formatted_data['X']),
                'input_shape': formatted_data['X'].shape,
                'target_shape': formatted_data['y'].shape,
                'feature_names': windows_result.get('feature_names', []),
                'step_size': self.step_size,
                'overlap_ratio': self.config.overlap_ratio,
                'data_completeness': self._calculate_completeness(formatted_data)
            }
            
            return FormattingResult(
                success=True,
                data=formatted_data,
                metadata=metadata,
                feature_names=windows_result.get('feature_names', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting time series: {e}")
            return FormattingResult(
                success=False,
                errors=[str(e)]
            )
    
    def _validate_input(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate input data."""
        errors = []
        
        # Check if data is empty
        if data.empty:
            errors.append("Input data is empty")
        
        # Check minimum length
        min_length = self.config.input_size + self.config.forecast_horizon
        if len(data) < min_length:
            errors.append(f"Data too short: {len(data)} < {min_length}")
        
        # Check target column
        if target_column not in data.columns:
            errors.append(f"Target column '{target_column}' not found")
        
        # Check feature columns
        if feature_columns:
            missing_features = set(feature_columns) - set(data.columns)
            if missing_features:
                errors.append(f"Missing feature columns: {missing_features}")
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Data must have DatetimeIndex")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Prepare data for window generation."""
        # Sort by index to ensure chronological order
        data_sorted = data.sort_index()
        
        # Extract target values
        target_values = data_sorted[target_column].values
        
        # Extract feature values
        if feature_columns:
            feature_values = data_sorted[feature_columns].values
            all_features = [target_column] + feature_columns
        else:
            feature_values = target_values.reshape(-1, 1)
            all_features = [target_column]
        
        # Convert to appropriate dtype
        if self.config.use_float32:
            target_values = target_values.astype(np.float32)
            feature_values = feature_values.astype(np.float32)
        
        return {
            'target_values': target_values,
            'feature_values': feature_values,
            'feature_names': all_features,
            'dates': data_sorted.index.values,
            'n_features': len(all_features)
        }
    
    def _generate_windows(
        self,
        prepared_data: Dict[str, Any],
        return_dates: bool = True
    ) -> Dict[str, Any]:
        """Generate sliding windows from prepared data."""
        try:
            target_values = prepared_data['target_values']
            feature_values = prepared_data['feature_values']
            dates = prepared_data['dates']
            
            n_samples = len(target_values)
            window_starts = []
            
            # Calculate window positions
            start_idx = 0
            while start_idx + self.config.input_size + self.config.forecast_horizon <= n_samples:
                window_starts.append(start_idx)
                start_idx += self.step_size
            
            if not window_starts:
                return {
                    'success': False,
                    'errors': ['No valid windows could be generated']
                }
            
            n_windows = len(window_starts)
            n_features = prepared_data['n_features']
            
            # Pre-allocate arrays
            dtype = np.float32 if self.config.use_float32 else np.float64
            
            X = np.zeros((n_windows, self.config.input_size, n_features), dtype=dtype)
            y = np.zeros((n_windows, self.config.forecast_horizon), dtype=dtype)
            
            if return_dates:
                window_dates = np.zeros((n_windows, self.config.input_size), dtype='datetime64[ns]')
                forecast_dates = np.zeros((n_windows, self.config.forecast_horizon), dtype='datetime64[ns]')
            
            # Fill arrays
            valid_windows = 0
            for i, start_idx in enumerate(window_starts):
                end_idx = start_idx + self.config.input_size
                forecast_end = end_idx + self.config.forecast_horizon
                
                # Extract input window
                input_window = feature_values[start_idx:end_idx]
                target_window = target_values[end_idx:forecast_end]
                
                # Check window completeness
                if self._check_window_completeness(input_window, target_window):
                    X[valid_windows] = input_window
                    y[valid_windows] = target_window
                    
                    if return_dates:
                        window_dates[valid_windows] = dates[start_idx:end_idx]
                        forecast_dates[valid_windows] = dates[end_idx:forecast_end]
                    
                    valid_windows += 1
            
            # Trim to valid windows
            if valid_windows < n_windows:
                X = X[:valid_windows]
                y = y[:valid_windows]
                if return_dates:
                    window_dates = window_dates[:valid_windows]
                    forecast_dates = forecast_dates[:valid_windows]
            
            result_data = {
                'X': X,
                'y': y
            }
            
            if return_dates:
                result_data['dates'] = window_dates
                result_data['forecast_dates'] = forecast_dates
            
            return {
                'success': True,
                'data': result_data,
                'feature_names': prepared_data['feature_names']
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Window generation failed: {str(e)}"]
            }
    
    async def _generate_windows_chunked(
        self,
        prepared_data: Dict[str, Any],
        return_dates: bool = True
    ) -> Dict[str, Any]:
        """Generate windows using chunked processing for memory efficiency."""
        target_values = prepared_data['target_values']
        n_samples = len(target_values)
        
        # Calculate total windows
        total_windows = 0
        start_idx = 0
        while start_idx + self.config.input_size + self.config.forecast_horizon <= n_samples:
            total_windows += 1
            start_idx += self.step_size
        
        if total_windows == 0:
            return {
                'success': False,
                'errors': ['No valid windows could be generated']
            }
        
        # Process in chunks
        chunks = []
        chunk_starts = list(range(0, total_windows, self.config.chunk_size))
        
        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + self.config.chunk_size, total_windows)
            
            # Create subset of data for this chunk
            window_subset_starts = []
            start_idx = 0
            for i in range(total_windows):
                if chunk_start <= i < chunk_end:
                    window_subset_starts.append(start_idx)
                start_idx += self.step_size
            
            # Generate windows for chunk
            chunk_data = prepared_data.copy()
            chunk_result = self._generate_windows_for_subset(
                chunk_data, window_subset_starts, return_dates
            )
            
            if chunk_result['success']:
                chunks.append(chunk_result['data'])
        
        # Combine chunks
        if chunks:
            combined_data = self._combine_chunks(chunks)
            return {
                'success': True,
                'data': combined_data,
                'feature_names': prepared_data['feature_names']
            }
        else:
            return {
                'success': False,
                'errors': ['Failed to generate any valid chunks']
            }
    
    def _generate_windows_for_subset(
        self,
        prepared_data: Dict[str, Any],
        window_starts: List[int],
        return_dates: bool
    ) -> Dict[str, Any]:
        """Generate windows for a subset of positions."""
        try:
            target_values = prepared_data['target_values']
            feature_values = prepared_data['feature_values']
            dates = prepared_data['dates']
            
            n_windows = len(window_starts)
            n_features = prepared_data['n_features']
            
            # Pre-allocate arrays
            dtype = np.float32 if self.config.use_float32 else np.float64
            
            X = np.zeros((n_windows, self.config.input_size, n_features), dtype=dtype)
            y = np.zeros((n_windows, self.config.forecast_horizon), dtype=dtype)
            
            if return_dates:
                window_dates = np.zeros((n_windows, self.config.input_size), dtype='datetime64[ns]')
                forecast_dates = np.zeros((n_windows, self.config.forecast_horizon), dtype='datetime64[ns]')
            
            # Fill arrays
            for i, start_idx in enumerate(window_starts):
                end_idx = start_idx + self.config.input_size
                forecast_end = end_idx + self.config.forecast_horizon
                
                X[i] = feature_values[start_idx:end_idx]
                y[i] = target_values[end_idx:forecast_end]
                
                if return_dates:
                    window_dates[i] = dates[start_idx:end_idx]
                    forecast_dates[i] = dates[end_idx:forecast_end]
            
            result_data = {
                'X': X,
                'y': y
            }
            
            if return_dates:
                result_data['dates'] = window_dates
                result_data['forecast_dates'] = forecast_dates
            
            return {
                'success': True,
                'data': result_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Subset window generation failed: {str(e)}"]
            }
    
    def _combine_chunks(self, chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine data from multiple chunks."""
        combined = {}
        
        for key in chunks[0].keys():
            arrays = [chunk[key] for chunk in chunks]
            combined[key] = np.concatenate(arrays, axis=0)
        
        return combined
    
    def _generate_multi_scale_windows(
        self,
        base_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate multi-scale windows for NHITS hierarchical processing."""
        multi_scale_data = {}
        
        base_X = base_data['X']
        base_y = base_data['y']
        
        for scale_factor in self.config.scale_factors:
            if scale_factor == 1:
                continue  # Skip base scale
            
            # Downsample input sequences
            downsampled_input_size = self.config.input_size // scale_factor
            downsampled_forecast_horizon = self.config.forecast_horizon // scale_factor
            
            if downsampled_input_size < 1 or downsampled_forecast_horizon < 1:
                continue
            
            # Average pooling for downsampling
            X_downsampled = self._average_pool_3d(
                base_X, pool_size=scale_factor, target_size=downsampled_input_size
            )
            y_downsampled = self._average_pool_2d(
                base_y, pool_size=scale_factor, target_size=downsampled_forecast_horizon
            )
            
            multi_scale_data[f'X_scale_{scale_factor}'] = X_downsampled
            multi_scale_data[f'y_scale_{scale_factor}'] = y_downsampled
        
        return multi_scale_data
    
    def _average_pool_3d(
        self,
        data: np.ndarray,
        pool_size: int,
        target_size: int
    ) -> np.ndarray:
        """Apply average pooling to 3D data (batch, time, features)."""
        batch_size, seq_len, n_features = data.shape
        
        # Reshape for pooling
        pooled_data = np.zeros((batch_size, target_size, n_features), dtype=data.dtype)
        
        for i in range(target_size):
            start_idx = i * pool_size
            end_idx = min(start_idx + pool_size, seq_len)
            pooled_data[:, i, :] = np.mean(data[:, start_idx:end_idx, :], axis=1)
        
        return pooled_data
    
    def _average_pool_2d(
        self,
        data: np.ndarray,
        pool_size: int,
        target_size: int
    ) -> np.ndarray:
        """Apply average pooling to 2D data (batch, time)."""
        batch_size, seq_len = data.shape
        
        # Reshape for pooling
        pooled_data = np.zeros((batch_size, target_size), dtype=data.dtype)
        
        for i in range(target_size):
            start_idx = i * pool_size
            end_idx = min(start_idx + pool_size, seq_len)
            pooled_data[:, i] = np.mean(data[:, start_idx:end_idx], axis=1)
        
        return pooled_data
    
    def _check_window_completeness(
        self,
        input_window: np.ndarray,
        target_window: np.ndarray
    ) -> bool:
        """Check if window meets completeness requirements."""
        # Check for NaN values
        input_completeness = 1 - np.isnan(input_window).sum() / input_window.size
        target_completeness = 1 - np.isnan(target_window).sum() / target_window.size
        
        min_completeness = self.config.min_window_completeness
        
        return (input_completeness >= min_completeness and 
                target_completeness >= min_completeness)
    
    def _calculate_completeness(self, data: Dict[str, np.ndarray]) -> float:
        """Calculate overall data completeness."""
        X = data['X']
        y = data['y']
        
        total_elements = X.size + y.size
        nan_elements = np.isnan(X).sum() + np.isnan(y).sum()
        
        return 1 - (nan_elements / total_elements)
    
    def format_for_inference(
        self,
        recent_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Format recent data for real-time inference.
        
        Args:
            recent_data: Recent time series data (last input_size points)
            feature_names: Names of features in the data
        
        Returns:
            Dictionary with formatted data ready for model inference
        """
        if len(recent_data) < self.config.input_size:
            raise ValueError(f"Need at least {self.config.input_size} data points")
        
        # Take the last input_size points
        input_data = recent_data[-self.config.input_size:]
        
        # Reshape for neural network (add batch dimension)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, self.config.input_size, 1)
        elif input_data.ndim == 2:
            input_data = input_data.reshape(1, self.config.input_size, -1)
        
        # Convert to appropriate dtype
        if self.config.use_float32:
            input_data = input_data.astype(np.float32)
        
        result = {'X': input_data}
        
        # Generate multi-scale inputs if enabled
        if self.config.enable_multi_scale:
            for scale_factor in self.config.scale_factors:
                if scale_factor == 1:
                    continue
                
                downsampled_size = self.config.input_size // scale_factor
                if downsampled_size < 1:
                    continue
                
                X_downsampled = self._average_pool_3d(
                    input_data, pool_size=scale_factor, target_size=downsampled_size
                )
                result[f'X_scale_{scale_factor}'] = X_downsampled
        
        return result
    
    def get_config(self) -> TimeSeriesConfig:
        """Get current configuration."""
        return self.config
    
    def get_window_info(self, data_length: int) -> Dict[str, int]:
        """Get information about windows that would be generated."""
        n_windows = 0
        start_idx = 0
        while start_idx + self.config.input_size + self.config.forecast_horizon <= data_length:
            n_windows += 1
            start_idx += self.step_size
        
        return {
            'n_windows': n_windows,
            'step_size': self.step_size,
            'input_size': self.config.input_size,
            'forecast_horizon': self.config.forecast_horizon,
            'overlap_samples': self.config.input_size - self.step_size,
            'coverage_ratio': (n_windows * self.step_size + self.config.input_size) / data_length if data_length > 0 else 0
        }