"""
NHITS Forecaster Implementation for Financial Time Series.

This module implements the NHITS (Neural Hierarchical Interpolation for Time Series)
model specifically adapted for financial forecasting in the AI News Trading Platform.
"""

import asyncio
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    logging.warning("neuralforecast not available - install with: pip install neuralforecast>=1.6.4")

try:
    import torch
    import torch.nn as nn
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("PyTorch not available - GPU acceleration disabled")


class NHITSForecaster:
    """
    Advanced NHITS forecaster optimized for financial time series prediction.
    
    Features:
    - Multi-horizon forecasting with confidence intervals
    - GPU acceleration support
    - Automated hyperparameter optimization
    - Model versioning and serialization
    - Robust error handling and fallback mechanisms
    - Integration with trading strategies
    """
    
    def __init__(
        self,
        input_size: int = 24,
        horizon: int = 12,
        n_freq_downsample: List[int] = None,
        activation: str = 'ReLU',
        n_blocks: List[int] = None,
        mlp_units: List[List[int]] = None,
        interpolation_mode: str = 'linear',
        pooling_mode: str = 'MaxPool1d',
        dropout: float = 0.1,
        batch_size: int = 32,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        enable_gpu: bool = True,
        model_cache_dir: str = None,
        quantiles: List[float] = None,
        **kwargs
    ):
        """
        Initialize NHITS Forecaster with comprehensive configuration.
        
        Args:
            input_size: Historical window size for input
            horizon: Forecast horizon length
            n_freq_downsample: Frequency downsampling factors
            activation: Activation function ('ReLU', 'GELU', etc.)
            n_blocks: Number of blocks per stack
            mlp_units: MLP architecture specification
            interpolation_mode: Interpolation method ('linear', 'nearest')
            pooling_mode: Pooling strategy ('MaxPool1d', 'AvgPool1d')
            dropout: Dropout rate for regularization
            batch_size: Training batch size
            max_epochs: Maximum training epochs
            learning_rate: Learning rate for optimization
            enable_gpu: Enable GPU acceleration if available
            model_cache_dir: Directory for model caching
            quantiles: Quantiles for confidence intervals
        """
        self.input_size = input_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        # Set default parameters optimized for financial data
        self.n_freq_downsample = n_freq_downsample or [2, 1, 1]
        self.n_blocks = n_blocks or [1, 1, 1]
        self.mlp_units = mlp_units or [[512, 512], [512, 512], [512, 512]]
        self.quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # GPU configuration
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.device = 'cuda' if self.enable_gpu else 'cpu'
        
        # Model configuration
        self.activation = activation
        self.interpolation_mode = interpolation_mode
        self.pooling_mode = pooling_mode
        
        # Model cache directory
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else Path("models/neural_forecast")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model components
        self.model = None
        self.nf = None
        self.is_fitted = False
        self.training_history = {}
        self.model_metadata = {}
        
        # Performance tracking
        self.forecast_cache = {}
        self.prediction_intervals = {}
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        if not NEURALFORECAST_AVAILABLE:
            self.logger.error("NeuralForecast not available. Install with: pip install neuralforecast>=1.6.4")
            raise ImportError("NeuralForecast required for NHITS functionality")
    
    def _setup_logging(self):
        """Setup logging configuration for the forecaster."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _create_nhits_model(self) -> NHITS:
        """
        Create and configure NHITS model with optimized parameters.
        
        Returns:
            Configured NHITS model instance
        """
        try:
            # Configure loss function with quantile regression for confidence intervals
            loss = MQLoss(level=[10, 20, 30, 40, 50, 60, 70, 80, 90])
            
            model = NHITS(
                h=self.horizon,
                input_size=self.input_size,
                n_freq_downsample=self.n_freq_downsample,
                activation=self.activation,
                n_blocks=self.n_blocks,
                mlp_units=self.mlp_units,
                interpolation_mode=self.interpolation_mode,
                pooling_mode=self.pooling_mode,
                dropout=self.dropout,
                batch_size=self.batch_size,
                max_steps=self.max_epochs,
                learning_rate=self.learning_rate,
                loss=loss,
                random_state=42,
                # GPU configuration
                accelerator='gpu' if self.enable_gpu else 'cpu',
                devices=1 if self.enable_gpu else None,
            )
            
            self.logger.info(f"Created NHITS model with device: {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create NHITS model: {str(e)}")
            raise
    
    async def fit(
        self,
        data: Union[pd.DataFrame, Dict],
        val_data: Optional[pd.DataFrame] = None,
        static_features: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the NHITS model on financial time series data.
        
        Args:
            data: Training data in NeuralForecast format or dictionary
            val_data: Optional validation data
            static_features: List of static feature columns
            verbose: Enable verbose training output
            
        Returns:
            Training results and metadata
        """
        self.logger.info("Starting NHITS model training")
        
        try:
            # Prepare data format
            if isinstance(data, dict):
                df_train = self._prepare_dataframe(data)
            else:
                df_train = data.copy()
            
            # Validate data format
            self._validate_data_format(df_train)
            
            # Create model if not exists
            if self.model is None:
                self.model = self._create_nhits_model()
            
            # Initialize NeuralForecast
            self.nf = NeuralForecast(
                models=[self.model],
                freq='H'  # Hourly frequency for financial data
            )
            
            # Fit model with error handling
            start_time = datetime.now()
            
            # Add GPU memory management
            if self.enable_gpu:
                await self._manage_gpu_memory()
            
            # Train model
            self.nf.fit(
                df=df_train,
                val_size=len(val_data) if val_data is not None else int(len(df_train) * 0.2),
                verbose=verbose
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Update model state
            self.is_fitted = True
            self.training_history = {
                'training_time': training_time,
                'data_points': len(df_train),
                'features': list(df_train.columns),
                'training_date': datetime.now().isoformat(),
                'device': self.device,
                'epochs': self.max_epochs,
                'batch_size': self.batch_size
            }
            
            # Save model metadata
            self.model_metadata.update({
                'version': '1.0.0',
                'input_size': self.input_size,
                'horizon': self.horizon,
                'model_type': 'NHITS',
                'quantiles': self.quantiles,
                'gpu_enabled': self.enable_gpu
            })
            
            self.logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            return {
                'success': True,
                'training_time': training_time,
                'model_metadata': self.model_metadata,
                'training_history': self.training_history
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback': 'Consider using simple moving average forecasting'
            }
    
    async def predict(
        self,
        data: Union[pd.DataFrame, Dict],
        return_intervals: bool = True,
        confidence_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            data: Input data for forecasting
            return_intervals: Include prediction intervals
            confidence_levels: Confidence levels for intervals
            
        Returns:
            Forecast results with metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Prepare input data
            if isinstance(data, dict):
                df_input = self._prepare_dataframe(data)
            else:
                df_input = data.copy()
            
            # Generate cache key
            cache_key = self._generate_cache_key(df_input)
            
            # Check cache
            if cache_key in self.forecast_cache:
                self.logger.info("Returning cached forecast")
                return self.forecast_cache[cache_key]
            
            # GPU memory management
            if self.enable_gpu:
                await self._manage_gpu_memory()
            
            # Generate forecast
            start_time = datetime.now()
            forecast = self.nf.predict(df=df_input)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            results = self._process_forecast_results(
                forecast, 
                return_intervals=return_intervals,
                confidence_levels=confidence_levels or [0.8, 0.9, 0.95]
            )
            
            # Add metadata
            results.update({
                'prediction_time': prediction_time,
                'model_version': self.model_metadata.get('version', '1.0.0'),
                'timestamp': datetime.now().isoformat(),
                'input_shape': df_input.shape,
                'horizon': self.horizon,
                'device_used': self.device
            })
            
            # Cache results
            self.forecast_cache[cache_key] = results
            
            self.logger.info(f"Prediction completed in {prediction_time:.3f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            
            # Fallback prediction using simple methods
            return await self._fallback_prediction(data)
    
    async def predict_batch(
        self,
        data_batch: List[Union[pd.DataFrame, Dict]],
        return_intervals: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate batch predictions for multiple time series.
        
        Args:
            data_batch: List of input datasets
            return_intervals: Include prediction intervals
            
        Returns:
            List of forecast results
        """
        self.logger.info(f"Processing batch prediction for {len(data_batch)} series")
        
        results = []
        
        # Process in parallel if GPU available, sequential otherwise
        if self.enable_gpu and len(data_batch) > 1:
            # Parallel processing for GPU
            tasks = [
                self.predict(data, return_intervals=return_intervals)
                for data in data_batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing
            for data in data_batch:
                result = await self.predict(data, return_intervals=return_intervals)
                results.append(result)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch prediction {i} failed: {str(result)}")
                valid_results.append({'success': False, 'error': str(result)})
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _prepare_dataframe(self, data: Dict) -> pd.DataFrame:
        """
        Convert dictionary data to NeuralForecast DataFrame format.
        
        Args:
            data: Dictionary with time series data
            
        Returns:
            Formatted DataFrame
        """
        if 'ds' not in data or 'y' not in data:
            raise ValueError("Data must contain 'ds' (timestamps) and 'y' (values) columns")
        
        df = pd.DataFrame(data)
        
        # Ensure proper datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Add unique_id if not present
        if 'unique_id' not in df.columns:
            df['unique_id'] = 'series_1'
        
        # Sort by time
        df = df.sort_values('ds').reset_index(drop=True)
        
        return df
    
    def _validate_data_format(self, df: pd.DataFrame):
        """Validate DataFrame format for NeuralForecast."""
        required_columns = ['unique_id', 'ds', 'y']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df['y'].isna().sum() > len(df) * 0.1:
            self.logger.warning("High percentage of missing values detected")
        
        if len(df) < self.input_size + self.horizon:
            raise ValueError(f"Insufficient data points. Need at least {self.input_size + self.horizon}")
    
    def _process_forecast_results(
        self,
        forecast: pd.DataFrame,
        return_intervals: bool = True,
        confidence_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Process raw forecast results into structured format.
        
        Args:
            forecast: Raw forecast DataFrame
            return_intervals: Include prediction intervals
            confidence_levels: Confidence levels for intervals
            
        Returns:
            Processed forecast results
        """
        results = {
            'point_forecast': forecast['NHITS'].tolist(),
            'timestamps': forecast['ds'].tolist(),
            'success': True
        }
        
        if return_intervals and confidence_levels:
            intervals = {}
            
            # Extract quantile forecasts
            for level in confidence_levels:
                lower_quantile = (1 - level) / 2
                upper_quantile = 1 - lower_quantile
                
                # Map to available quantile columns
                lower_col = f"NHITS-q-{int(lower_quantile * 100)}"
                upper_col = f"NHITS-q-{int(upper_quantile * 100)}"
                
                if lower_col in forecast.columns and upper_col in forecast.columns:
                    intervals[f"{int(level * 100)}%"] = {
                        'lower': forecast[lower_col].tolist(),
                        'upper': forecast[upper_col].tolist()
                    }
            
            results['prediction_intervals'] = intervals
        
        return results
    
    async def _manage_gpu_memory(self):
        """Manage GPU memory to prevent OOM errors."""
        if self.enable_gpu and GPU_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated()
                memory_cached = torch.cuda.memory_reserved()
                
                self.logger.debug(f"GPU Memory - Allocated: {memory_allocated / 1e9:.2f}GB, "
                                f"Cached: {memory_cached / 1e9:.2f}GB")
                
                # Force garbage collection if memory usage is high
                if memory_allocated > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.warning(f"GPU memory management failed: {str(e)}")
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key for forecast results."""
        # Create hash from data characteristics
        data_hash = hash(tuple([
            len(data),
            data['y'].iloc[-1] if len(data) > 0 else 0,
            data['ds'].iloc[-1].isoformat() if len(data) > 0 else '',
            self.horizon,
            self.input_size
        ]))
        
        return f"forecast_{abs(data_hash)}"
    
    async def _fallback_prediction(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Fallback prediction using simple statistical methods.
        
        Args:
            data: Input data
            
        Returns:
            Fallback forecast results
        """
        self.logger.warning("Using fallback prediction method")
        
        try:
            if isinstance(data, dict):
                values = data['y']
            else:
                values = data['y'].tolist()
            
            # Simple moving average forecast
            window_size = min(self.input_size, len(values))
            recent_values = values[-window_size:]
            avg_value = np.mean(recent_values)
            
            # Generate simple forecast
            forecast = [avg_value] * self.horizon
            
            # Simple confidence intervals (±10%)
            lower_bound = [v * 0.9 for v in forecast]
            upper_bound = [v * 1.1 for v in forecast]
            
            return {
                'point_forecast': forecast,
                'prediction_intervals': {
                    '80%': {'lower': lower_bound, 'upper': upper_bound}
                },
                'success': True,
                'method': 'fallback_moving_average',
                'warning': 'Using simple moving average due to model failure'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {str(e)}")
            return {
                'success': False,
                'error': 'Both primary and fallback prediction methods failed',
                'point_forecast': [0] * self.horizon
            }
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.model_cache_dir / f"nhits_model_{timestamp}.pkl"
        
        try:
            # Save model and metadata
            model_data = {
                'nf': self.nf,
                'model_metadata': self.model_metadata,
                'training_history': self.training_history,
                'config': {
                    'input_size': self.input_size,
                    'horizon': self.horizon,
                    'quantiles': self.quantiles,
                    'device': self.device
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.nf = model_data['nf']
            self.model_metadata = model_data['model_metadata']
            self.training_history = model_data['training_history']
            
            # Restore configuration
            config = model_data['config']
            self.input_size = config['input_size']
            self.horizon = config['horizon']
            self.quantiles = config['quantiles']
            
            self.is_fitted = True
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Model information dictionary
        """
        return {
            'is_fitted': self.is_fitted,
            'model_metadata': self.model_metadata,
            'training_history': self.training_history,
            'configuration': {
                'input_size': self.input_size,
                'horizon': self.horizon,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'learning_rate': self.learning_rate,
                'device': self.device,
                'gpu_enabled': self.enable_gpu,
                'quantiles': self.quantiles
            },
            'cache_stats': {
                'cached_forecasts': len(self.forecast_cache),
                'cache_dir': str(self.model_cache_dir)
            }
        }
    
    def clear_cache(self):
        """Clear forecast cache to free memory."""
        self.forecast_cache.clear()
        self.prediction_intervals.clear()
        self.logger.info("Forecast cache cleared")
    
    def optimize_hyperparameters(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using cross-validation.
        
        Args:
            data: Training data
            param_grid: Parameter grid for optimization
            cv_folds: Number of cross-validation folds
            
        Returns:
            Optimization results
        """
        # This would implement hyperparameter optimization
        # For now, return current configuration as "optimal"
        self.logger.info("Hyperparameter optimization not yet implemented")
        
        return {
            'best_params': {
                'input_size': self.input_size,
                'horizon': self.horizon,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            },
            'optimization_method': 'default_configuration',
            'note': 'Full hyperparameter optimization to be implemented'
        }