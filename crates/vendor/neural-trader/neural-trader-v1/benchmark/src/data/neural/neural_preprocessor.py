"""
Neural Data Preprocessor - Main orchestrator for neural forecasting data pipeline.

This module coordinates all data preprocessing steps required for NHITS and other
neural forecasting models, ensuring data is properly formatted, cleaned, and optimized
for neural network training and inference.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, AsyncIterator
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .time_series_formatter import TimeSeriesFormatter, TimeSeriesConfig
from .missing_value_imputer import MissingValueImputer, ImputationConfig
from .outlier_detector import OutlierDetector, OutlierConfig
from .feature_engineer import FeatureEngineer, FeatureConfig
from .data_normalizer import DataNormalizer, NormalizationConfig
from .data_augmenter import DataAugmenter, AugmentationConfig
from .version_manager import DataVersionManager
from .quality_monitor import NeuralDataQualityMonitor

logger = logging.getLogger(__name__)


@dataclass
class NeuralPreprocessingConfig:
    """Configuration for neural data preprocessing pipeline."""
    # Model configuration
    input_size: int = 480  # Lookback window
    forecast_horizon: int = 96  # Forecast horizon
    batch_size: int = 256
    
    # Time series configuration
    frequency: str = "5min"  # Data frequency
    seasonality: Optional[List[int]] = None  # [24, 168] for hourly with daily/weekly
    
    # Processing configuration
    handle_missing_values: bool = True
    detect_outliers: bool = True
    apply_feature_engineering: bool = True
    normalize_data: bool = True
    augment_data: bool = False  # Only for training
    
    # Quality thresholds
    min_data_completeness: float = 0.8  # Minimum 80% data completeness
    max_outlier_ratio: float = 0.05  # Maximum 5% outliers
    
    # Performance settings
    use_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    
    # Storage settings
    enable_versioning: bool = True
    enable_caching: bool = True
    cache_dir: str = "neural_cache"
    
    def __post_init__(self):
        if self.seasonality is None:
            # Default seasonality patterns based on frequency
            if self.frequency == "5min":
                self.seasonality = [288, 2016]  # Daily (288) and weekly (2016) for 5-min data
            elif self.frequency == "1h":
                self.seasonality = [24, 168]  # Daily and weekly for hourly data
            else:
                self.seasonality = [1]  # No seasonality


@dataclass
class ProcessingResult:
    """Result of neural data preprocessing."""
    success: bool
    data: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetSplit:
    """Dataset split for training/validation/testing."""
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: Optional[np.ndarray] = None
    val_y: Optional[np.ndarray] = None
    test_x: Optional[np.ndarray] = None
    test_y: Optional[np.ndarray] = None
    
    # Metadata
    train_dates: Optional[np.ndarray] = None
    val_dates: Optional[np.ndarray] = None
    test_dates: Optional[np.ndarray] = None
    
    # Normalization parameters
    scaler_params: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    data_quality: Dict[str, float] = field(default_factory=dict)


class NeuralDataPreprocessor:
    """
    Main orchestrator for neural forecasting data preprocessing.
    
    Features:
    - Time series formatting with sliding windows
    - Missing value imputation
    - Outlier detection and handling
    - Feature engineering
    - Data normalization
    - Data augmentation (for training)
    - Quality monitoring and validation
    - Data versioning and caching
    - Multi-symbol processing
    - Real-time processing support
    """
    
    def __init__(self, config: NeuralPreprocessingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.time_series_formatter = TimeSeriesFormatter(
            TimeSeriesConfig(
                input_size=config.input_size,
                forecast_horizon=config.forecast_horizon,
                frequency=config.frequency,
                seasonality=config.seasonality
            )
        )
        
        self.missing_value_imputer = MissingValueImputer(
            ImputationConfig(
                strategy='forward_fill',
                max_gap_size=12,  # 1 hour for 5-min data
                min_valid_ratio=config.min_data_completeness
            )
        ) if config.handle_missing_values else None
        
        self.outlier_detector = OutlierDetector(
            OutlierConfig(
                method='statistical',
                z_score_threshold=3.0,
                max_outlier_ratio=config.max_outlier_ratio
            )
        ) if config.detect_outliers else None
        
        self.feature_engineer = FeatureEngineer(
            FeatureConfig(
                add_technical_indicators=True,
                add_temporal_features=True,
                add_lag_features=True,
                lookback_periods=[1, 5, 12, 24]
            )
        ) if config.apply_feature_engineering else None
        
        self.data_normalizer = DataNormalizer(
            NormalizationConfig(
                method='robust',
                feature_range=(-1, 1),
                per_feature_scaling=True
            )
        ) if config.normalize_data else None
        
        self.data_augmenter = DataAugmenter(
            AugmentationConfig(
                noise_ratio=0.01,
                jitter_ratio=0.02,
                time_warping=True,
                magnitude_warping=True
            )
        ) if config.augment_data else None
        
        # Management components
        self.version_manager = DataVersionManager() if config.enable_versioning else None
        self.quality_monitor = NeuralDataQualityMonitor()
        
        # Processing state
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.processing_stats = defaultdict(list)
        self.processing_history = deque(maxlen=1000)
        
        self.logger.info("Neural Data Preprocessor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the preprocessor."""
        logger = logging.getLogger('neural_preprocessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def preprocess_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None,
        for_training: bool = True
    ) -> ProcessingResult:
        """
        Preprocess data for a single symbol.
        
        Args:
            symbol: Symbol identifier
            data: Input DataFrame with time series data
            target_column: Column to forecast
            feature_columns: Additional feature columns
            for_training: Whether data is for training (enables augmentation)
        
        Returns:
            ProcessingResult with processed data and metadata
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            self.logger.info(f"Preprocessing {symbol}: {len(data)} samples")
            
            # Step 1: Data validation
            validation_result = await self._validate_input_data(data, target_column)
            if not validation_result['valid']:
                return ProcessingResult(
                    success=False,
                    errors=validation_result['errors']
                )
            
            processed_data = data.copy()
            
            # Step 2: Handle missing values
            if self.missing_value_imputer:
                imputation_result = await self.missing_value_imputer.impute(
                    processed_data, target_column
                )
                if imputation_result.success:
                    processed_data = imputation_result.data
                    if imputation_result.warnings:
                        warnings.extend(imputation_result.warnings)
                else:
                    errors.extend(imputation_result.errors)
            
            # Step 3: Detect and handle outliers
            if self.outlier_detector:
                outlier_result = await self.outlier_detector.detect_and_handle(
                    processed_data, target_column
                )
                if outlier_result.success:
                    processed_data = outlier_result.data
                    if outlier_result.warnings:
                        warnings.extend(outlier_result.warnings)
                else:
                    errors.extend(outlier_result.errors)
            
            # Step 4: Feature engineering
            if self.feature_engineer:
                feature_result = await self.feature_engineer.engineer_features(
                    processed_data, target_column, feature_columns
                )
                if feature_result.success:
                    processed_data = feature_result.data
                    feature_columns = feature_result.feature_names
                else:
                    errors.extend(feature_result.errors)
            
            # Step 5: Data normalization
            scaler_params = None
            if self.data_normalizer:
                normalization_result = await self.data_normalizer.normalize(
                    processed_data, target_column, feature_columns
                )
                if normalization_result.success:
                    processed_data = normalization_result.data
                    scaler_params = normalization_result.scaler_params
                else:
                    errors.extend(normalization_result.errors)
            
            # Step 6: Format time series
            formatting_result = await self.time_series_formatter.format_for_neural_network(
                processed_data, target_column, feature_columns
            )
            if not formatting_result.success:
                errors.extend(formatting_result.errors)
                return ProcessingResult(success=False, errors=errors)
            
            formatted_data = formatting_result.data
            
            # Step 7: Data augmentation (training only)
            if for_training and self.data_augmenter:
                augmentation_result = await self.data_augmenter.augment(formatted_data)
                if augmentation_result.success:
                    formatted_data.update(augmentation_result.data)
                else:
                    warnings.extend(augmentation_result.warnings)
            
            # Step 8: Quality assessment
            quality_metrics = await self.quality_monitor.assess_quality(
                formatted_data, processed_data
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store processing statistics
            self.processing_stats[symbol].append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'input_samples': len(data),
                'output_samples': len(formatted_data.get('X', [])),
                'quality_score': quality_metrics.get('overall_quality', 0.0)
            })
            
            # Create result
            result = ProcessingResult(
                success=len(errors) == 0,
                data=formatted_data,
                metadata={
                    'symbol': symbol,
                    'input_samples': len(data),
                    'output_samples': len(formatted_data.get('X', [])),
                    'feature_columns': feature_columns,
                    'scaler_params': scaler_params,
                    'processing_steps': self._get_processing_steps()
                },
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
            
            # Cache result if enabled
            if self.config.enable_caching:
                await self._cache_result(symbol, result)
            
            # Version control
            if self.version_manager:
                await self.version_manager.track_processing(symbol, result)
            
            self.logger.info(
                f"Preprocessed {symbol}: {len(formatted_data.get('X', []))} windows "
                f"in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {symbol}: {e}")
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time
            )
    
    async def preprocess_multiple_symbols(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None,
        for_training: bool = True
    ) -> Dict[str, ProcessingResult]:
        """
        Preprocess data for multiple symbols in parallel.
        
        Args:
            symbol_data: Dictionary mapping symbols to DataFrames
            target_column: Column to forecast
            feature_columns: Additional feature columns
            for_training: Whether data is for training
        
        Returns:
            Dictionary mapping symbols to ProcessingResults
        """
        if not self.config.use_parallel_processing:
            # Sequential processing
            results = {}
            for symbol, data in symbol_data.items():
                results[symbol] = await self.preprocess_symbol(
                    symbol, data, target_column, feature_columns, for_training
                )
            return results
        
        # Parallel processing
        tasks = []
        for symbol, data in symbol_data.items():
            task = asyncio.create_task(
                self.preprocess_symbol(
                    symbol, data, target_column, feature_columns, for_training
                )
            )
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                results[symbol] = await task
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = ProcessingResult(
                    success=False,
                    errors=[str(e)]
                )
        
        return results
    
    async def create_dataset_splits(
        self,
        processed_data: Dict[str, np.ndarray],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        time_ordered: bool = True
    ) -> DatasetSplit:
        """
        Create train/validation/test splits from processed data.
        
        Args:
            processed_data: Processed data from preprocessing
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            time_ordered: Whether to maintain time order in splits
        
        Returns:
            DatasetSplit with train/val/test data
        """
        X = processed_data['X']
        y = processed_data['y']
        dates = processed_data.get('dates')
        
        n_samples = len(X)
        
        if time_ordered:
            # Time-ordered splits
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_x, train_y = X[:train_end], y[:train_end]
            val_x, val_y = X[train_end:val_end], y[train_end:val_end]
            test_x, test_y = X[val_end:], y[val_end:]
            
            train_dates = dates[:train_end] if dates is not None else None
            val_dates = dates[train_end:val_end] if dates is not None else None
            test_dates = dates[val_end:] if dates is not None else None
        else:
            # Random splits
            indices = np.random.permutation(n_samples)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
            
            train_x, train_y = X[train_idx], y[train_idx]
            val_x, val_y = X[val_idx], y[val_idx]
            test_x, test_y = X[test_idx], y[test_idx]
            
            train_dates = dates[train_idx] if dates is not None else None
            val_dates = dates[val_idx] if dates is not None else None
            test_dates = dates[test_idx] if dates is not None else None
        
        # Calculate quality metrics for each split
        quality_metrics = {
            'train_samples': len(train_x),
            'val_samples': len(val_x) if val_x is not None else 0,
            'test_samples': len(test_x) if test_x is not None else 0,
            'total_samples': n_samples,
            'train_ratio_actual': len(train_x) / n_samples,
            'val_ratio_actual': len(val_x) / n_samples if val_x is not None else 0,
            'test_ratio_actual': len(test_x) / n_samples if test_x is not None else 0
        }
        
        return DatasetSplit(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            scaler_params=processed_data.get('scaler_params'),
            data_quality=quality_metrics
        )
    
    async def _validate_input_data(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Validate input data."""
        errors = []
        
        # Check if data is empty
        if data.empty:
            errors.append("Input data is empty")
        
        # Check if target column exists
        if target_column not in data.columns:
            errors.append(f"Target column '{target_column}' not found in data")
        
        # Check minimum data requirements
        min_samples = self.config.input_size + self.config.forecast_horizon
        if len(data) < min_samples:
            errors.append(
                f"Insufficient data: {len(data)} samples, need at least {min_samples}"
            )
        
        # Check for time index
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Data must have DatetimeIndex")
        
        # Check data completeness
        if target_column in data.columns:
            completeness = 1 - data[target_column].isna().sum() / len(data)
            if completeness < self.config.min_data_completeness:
                errors.append(
                    f"Data completeness {completeness:.2%} below threshold "
                    f"{self.config.min_data_completeness:.2%}"
                )
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _get_processing_steps(self) -> List[str]:
        """Get list of enabled processing steps."""
        steps = ['data_validation', 'time_series_formatting']
        
        if self.missing_value_imputer:
            steps.append('missing_value_imputation')
        if self.outlier_detector:
            steps.append('outlier_detection')
        if self.feature_engineer:
            steps.append('feature_engineering')
        if self.data_normalizer:
            steps.append('data_normalization')
        if self.data_augmenter:
            steps.append('data_augmentation')
        
        steps.append('quality_assessment')
        return steps
    
    async def _cache_result(self, symbol: str, result: ProcessingResult):
        """Cache preprocessing result."""
        cache_file = self.cache_dir / f"{symbol}_preprocessed.json"
        
        cache_data = {
            'symbol': symbol,
            'timestamp': time.time(),
            'success': result.success,
            'metadata': result.metadata,
            'quality_metrics': result.quality_metrics,
            'processing_time': result.processing_time,
            'errors': result.errors,
            'warnings': result.warnings
        }
        
        # Don't cache the actual data arrays (too large), just metadata
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        all_stats = []
        for symbol_stats in self.processing_stats.values():
            all_stats.extend(symbol_stats)
        
        if not all_stats:
            return {}
        
        processing_times = [s['processing_time'] for s in all_stats]
        quality_scores = [s['quality_score'] for s in all_stats]
        
        return {
            'total_processed': len(all_stats),
            'symbols_processed': len(self.processing_stats),
            'avg_processing_time': np.mean(processing_times),
            'median_processing_time': np.median(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'processing_rate': len(all_stats) / sum(processing_times) if sum(processing_times) > 0 else 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the preprocessor."""
        return {
            'config': {
                'input_size': self.config.input_size,
                'forecast_horizon': self.config.forecast_horizon,
                'frequency': self.config.frequency,
                'enabled_components': self._get_processing_steps()
            },
            'statistics': self.get_processing_statistics(),
            'cache_enabled': self.config.enable_caching,
            'versioning_enabled': self.config.enable_versioning,
            'parallel_processing': self.config.use_parallel_processing
        }