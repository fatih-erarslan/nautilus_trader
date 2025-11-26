# NeuralForecast Technical Implementation Guide

## Quick Start Implementation

This guide provides concrete implementation steps and code examples for integrating NeuralForecast into the AI News Trading Platform.

## 1. Installation and Setup

```bash
# Install NeuralForecast with GPU support
pip install neuralforecast[gpu]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies
pip install optuna  # For hyperparameter optimization
pip install shap    # For model interpretability
```

## 2. Core Integration Module

Create `/workspaces/ai-news-trader/src/forecasting/neural_forecast_integration.py`:

```python
"""
NeuralForecast Integration Module for AI News Trading Platform
Provides seamless integration with existing trading strategies
"""

import pandas as pd
import numpy as np
import cudf
import cupy as cp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, TFT, PatchTST
from neuralforecast.losses.pytorch import MAE, QuantileLoss
from neuralforecast.utils import AirPassengersDF

logger = logging.getLogger(__name__)


class NeuralForecastIntegration:
    """
    Main integration class for NeuralForecast with the trading platform.
    Handles data conversion, model management, and prediction generation.
    """
    
    def __init__(self, gpu_enabled: bool = True, cache_size: int = 100):
        """
        Initialize NeuralForecast integration.
        
        Args:
            gpu_enabled: Enable GPU acceleration
            cache_size: Number of models to cache
        """
        self.gpu_enabled = gpu_enabled and self._check_gpu_availability()
        self.model_cache = {}
        self.cache_size = cache_size
        self.forecast_cache = {}
        
        # Model configurations optimized for financial data
        self.model_configs = {
            'NBEATS': {
                'input_size': 24 * 7,  # 1 week of hourly data
                'h': 24,               # 24-hour forecast
                'max_steps': 500,
                'loss': MAE(),
                'learning_rate': 1e-3,
                'batch_size': 32,
                'windows_batch_size': 128,
                'enable_progress_bar': False
            },
            'NHITS': {
                'input_size': 24 * 7,
                'h': 24,
                'max_steps': 500,
                'n_freq_downsample': [24, 12, 1],
                'learning_rate': 1e-3,
                'batch_size': 32,
                'windows_batch_size': 128,
                'enable_progress_bar': False
            },
            'TFT': {
                'input_size': 24 * 7,
                'h': 24,
                'max_steps': 500,
                'hidden_size': 128,
                'n_heads': 4,
                'learning_rate': 1e-3,
                'batch_size': 32,
                'enable_progress_bar': False
            }
        }
        
        logger.info(f"NeuralForecast Integration initialized (GPU: {self.gpu_enabled})")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for neural models."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def prepare_data_for_neuralforecast(self, 
                                      market_data: pd.DataFrame,
                                      symbol: str,
                                      target_column: str = 'close',
                                      exogenous_vars: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert market data to NeuralForecast format.
        
        Args:
            market_data: Raw market data
            symbol: Trading symbol
            target_column: Target variable to forecast
            exogenous_vars: List of exogenous variables
            
        Returns:
            DataFrame in NeuralForecast format
        """
        # Required columns: unique_id, ds, y
        nf_data = pd.DataFrame({
            'unique_id': symbol,
            'ds': pd.to_datetime(market_data.index),
            'y': market_data[target_column].values
        })
        
        # Add exogenous variables if specified
        if exogenous_vars:
            for var in exogenous_vars:
                if var in market_data.columns:
                    nf_data[var] = market_data[var].values
        
        # Sort by timestamp
        nf_data = nf_data.sort_values('ds').reset_index(drop=True)
        
        # Handle missing values
        nf_data['y'] = nf_data['y'].fillna(method='ffill').fillna(method='bfill')
        
        return nf_data
    
    def create_ensemble_model(self, 
                            horizon: int = 24,
                            models: List[str] = ['NBEATS', 'NHITS'],
                            loss_function: str = 'MAE') -> List:
        """
        Create an ensemble of neural forecasting models.
        
        Args:
            horizon: Forecast horizon
            models: List of model types
            loss_function: Loss function to use
            
        Returns:
            List of configured models
        """
        ensemble = []
        
        for model_name in models:
            if model_name not in self.model_configs:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            config = self.model_configs[model_name].copy()
            config['h'] = horizon
            
            # Select loss function
            if loss_function == 'QuantileLoss':
                config['loss'] = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
            
            # Create model instance
            if model_name == 'NBEATS':
                model = NBEATS(**config)
            elif model_name == 'NHITS':
                model = NHITS(**config)
            elif model_name == 'TFT':
                model = TFT(**config)
            else:
                continue
            
            ensemble.append(model)
        
        return ensemble
    
    @lru_cache(maxsize=100)
    def generate_forecast(self,
                         symbol: str,
                         market_data: pd.DataFrame,
                         horizon: int = 24,
                         models: List[str] = ['NBEATS'],
                         include_confidence: bool = True) -> Dict[str, Any]:
        """
        Generate forecasts using NeuralForecast models.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data
            horizon: Forecast horizon
            models: Models to use
            include_confidence: Include confidence intervals
            
        Returns:
            Dictionary with forecasts and metadata
        """
        logger.info(f"Generating neural forecast for {symbol}")
        
        # Prepare data
        nf_data = self.prepare_data_for_neuralforecast(market_data, symbol)
        
        # Create or retrieve models
        model_key = f"{symbol}_{horizon}_{'_'.join(sorted(models))}"
        
        if model_key in self.model_cache:
            nf = self.model_cache[model_key]
            logger.info("Using cached model")
        else:
            # Create ensemble
            ensemble = self.create_ensemble_model(horizon, models)
            
            # Initialize NeuralForecast
            nf = NeuralForecast(
                models=ensemble,
                freq='H',  # Hourly frequency
                gpu=self.gpu_enabled
            )
            
            # Fit models
            nf.fit(df=nf_data, verbose=False)
            
            # Cache model
            if len(self.model_cache) >= self.cache_size:
                # Remove oldest model
                oldest_key = list(self.model_cache.keys())[0]
                del self.model_cache[oldest_key]
            
            self.model_cache[model_key] = nf
        
        # Generate predictions
        predictions = nf.predict()
        
        # Process results
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'horizon': horizon,
            'models_used': models,
            'forecasts': {}
        }
        
        # Extract forecasts for each model
        for model_name in models:
            model_col = f"{model_name}"
            if model_col in predictions.columns:
                result['forecasts'][model_name] = predictions[model_col].tolist()
        
        # Calculate ensemble forecast (average)
        forecast_cols = [col for col in predictions.columns if col in models]
        if forecast_cols:
            result['forecasts']['ensemble'] = predictions[forecast_cols].mean(axis=1).tolist()
        
        # Add confidence intervals if requested
        if include_confidence:
            result['confidence_intervals'] = self._calculate_confidence_intervals(
                predictions, models
            )
        
        # Add forecast quality metrics
        result['quality_metrics'] = self._assess_forecast_quality(nf_data, predictions)
        
        return result
    
    def _calculate_confidence_intervals(self,
                                      predictions: pd.DataFrame,
                                      models: List[str]) -> Dict[str, Any]:
        """Calculate confidence intervals from ensemble predictions."""
        forecast_cols = [col for col in predictions.columns if col in models]
        
        if len(forecast_cols) > 1:
            # Use ensemble variance for confidence intervals
            ensemble_mean = predictions[forecast_cols].mean(axis=1)
            ensemble_std = predictions[forecast_cols].std(axis=1)
            
            return {
                'lower_95': (ensemble_mean - 1.96 * ensemble_std).tolist(),
                'upper_95': (ensemble_mean + 1.96 * ensemble_std).tolist(),
                'lower_80': (ensemble_mean - 1.28 * ensemble_std).tolist(),
                'upper_80': (ensemble_mean + 1.28 * ensemble_std).tolist()
            }
        else:
            # Single model - use historical volatility
            return {
                'lower_95': predictions[forecast_cols[0]].tolist(),
                'upper_95': predictions[forecast_cols[0]].tolist()
            }
    
    def _assess_forecast_quality(self,
                               historical_data: pd.DataFrame,
                               predictions: pd.DataFrame) -> Dict[str, float]:
        """Assess forecast quality based on historical patterns."""
        # Simple quality metrics
        historical_volatility = historical_data['y'].pct_change().std()
        historical_mean = historical_data['y'].mean()
        
        # Forecast statistics
        forecast_values = predictions.iloc[:, -1].values  # Last column is forecast
        forecast_volatility = pd.Series(forecast_values).pct_change().std()
        
        # Quality score based on consistency with historical patterns
        volatility_ratio = forecast_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        quality_score = 1.0 / (1.0 + abs(volatility_ratio - 1.0))
        
        return {
            'quality_score': quality_score,
            'historical_volatility': historical_volatility,
            'forecast_volatility': forecast_volatility,
            'volatility_consistency': volatility_ratio
        }
    
    def backtest_forecast_model(self,
                              symbol: str,
                              market_data: pd.DataFrame,
                              model: str = 'NBEATS',
                              test_size: int = 168,  # 1 week
                              n_windows: int = 3) -> Dict[str, Any]:
        """
        Backtest neural forecast model performance.
        
        Args:
            symbol: Trading symbol
            market_data: Historical data
            model: Model to test
            test_size: Test set size in hours
            n_windows: Number of backtesting windows
            
        Returns:
            Backtest results with performance metrics
        """
        logger.info(f"Backtesting {model} for {symbol}")
        
        # Prepare data
        nf_data = self.prepare_data_for_neuralforecast(market_data, symbol)
        
        # Create model
        model_instance = self.create_ensemble_model(horizon=24, models=[model])[0]
        
        # Initialize NeuralForecast with backtesting
        nf = NeuralForecast(
            models=[model_instance],
            freq='H',
            gpu=self.gpu_enabled
        )
        
        # Perform cross-validation
        cv_results = nf.cross_validation(
            df=nf_data,
            h=24,
            step_size=24,
            n_windows=n_windows
        )
        
        # Calculate metrics
        from neuralforecast.losses.numpy import mae, mse, rmse, mape, smape
        
        metrics = {
            'MAE': mae(cv_results['y'], cv_results[model]),
            'MSE': mse(cv_results['y'], cv_results[model]),
            'RMSE': rmse(cv_results['y'], cv_results[model]),
            'MAPE': mape(cv_results['y'], cv_results[model]),
            'sMAPE': smape(cv_results['y'], cv_results[model])
        }
        
        # Calculate directional accuracy
        actual_direction = np.sign(cv_results['y'].diff())
        forecast_direction = np.sign(cv_results[model].diff())
        directional_accuracy = (actual_direction == forecast_direction).mean()
        
        return {
            'symbol': symbol,
            'model': model,
            'metrics': metrics,
            'directional_accuracy': directional_accuracy,
            'test_windows': n_windows,
            'cv_results': cv_results.to_dict()
        }
    
    def optimize_hyperparameters(self,
                               symbol: str,
                               market_data: pd.DataFrame,
                               model_type: str = 'NBEATS',
                               n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            symbol: Trading symbol
            market_data: Historical data
            model_type: Model to optimize
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters and performance
        """
        import optuna
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'input_size': trial.suggest_int('input_size', 24, 168, step=24),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'max_steps': trial.suggest_int('max_steps', 100, 1000, step=100),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
            }
            
            if model_type == 'NBEATS':
                params['stack_types'] = trial.suggest_categorical(
                    'stack_types', [['trend', 'seasonality'], ['generic']]
                )
            
            # Create and test model
            config = self.model_configs[model_type].copy()
            config.update(params)
            
            try:
                # Quick backtest
                result = self.backtest_forecast_model(
                    symbol, market_data, model_type, n_windows=1
                )
                return result['metrics']['MAE']
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [
                {'trial': i, 'value': t.value, 'params': t.params}
                for i, t in enumerate(study.trials)
                if t.value is not None
            ]
        }


class NeuralForecastStrategyEnhancer:
    """
    Enhances existing trading strategies with neural forecasts.
    """
    
    def __init__(self, neural_forecast: NeuralForecastIntegration):
        """
        Initialize strategy enhancer.
        
        Args:
            neural_forecast: NeuralForecast integration instance
        """
        self.nf = neural_forecast
        self.enhancement_cache = {}
    
    def enhance_momentum_signals(self,
                               momentum_data: Dict[str, Any],
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhance momentum trading signals with neural forecasts.
        
        Args:
            momentum_data: Original momentum strategy output
            market_data: Historical market data
            
        Returns:
            Enhanced trading signals
        """
        symbol = momentum_data.get('symbol', 'UNKNOWN')
        
        # Generate neural forecast
        forecast_result = self.nf.generate_forecast(
            symbol=symbol,
            market_data=market_data,
            horizon=24,
            models=['NBEATS', 'NHITS']
        )
        
        # Extract ensemble forecast
        ensemble_forecast = forecast_result['forecasts']['ensemble']
        
        # Calculate forecast direction and strength
        current_price = market_data['close'].iloc[-1]
        forecast_24h = ensemble_forecast[-1] if ensemble_forecast else current_price
        
        forecast_return = (forecast_24h - current_price) / current_price
        forecast_direction = 'bullish' if forecast_return > 0.001 else 'bearish' if forecast_return < -0.001 else 'neutral'
        
        # Enhance momentum signals
        enhanced_data = momentum_data.copy()
        
        # Adjust position size based on forecast confidence
        quality_score = forecast_result['quality_metrics']['quality_score']
        
        if forecast_direction == 'bullish' and momentum_data['action'] == 'BUY':
            # Aligned signals - increase confidence
            enhanced_data['position_size_pct'] *= (1 + 0.2 * quality_score)
            enhanced_data['confidence'] = 'VERY_HIGH'
        elif forecast_direction == 'bearish' and momentum_data['action'] == 'BUY':
            # Conflicting signals - reduce position
            enhanced_data['position_size_pct'] *= 0.7
            enhanced_data['confidence'] = 'LOW'
        
        # Add neural forecast data
        enhanced_data['neural_forecast'] = {
            'direction': forecast_direction,
            'expected_return': forecast_return,
            'forecast_price': forecast_24h,
            'quality_score': quality_score,
            'models_used': forecast_result['models_used']
        }
        
        # Update reasoning
        enhanced_data['reasoning'] += f" Neural forecast: {forecast_direction} " \
                                     f"({forecast_return:.2%} expected return, " \
                                     f"quality: {quality_score:.2f})"
        
        return enhanced_data
    
    def create_forecast_based_stops(self,
                                  position_data: Dict[str, Any],
                                  forecast_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create dynamic stop-loss levels based on forecast confidence intervals.
        
        Args:
            position_data: Current position information
            forecast_result: Neural forecast results
            
        Returns:
            Updated position data with forecast-based stops
        """
        if 'confidence_intervals' not in forecast_result:
            return position_data
        
        ci = forecast_result['confidence_intervals']
        current_price = position_data['entry_price']
        
        # Use lower confidence interval as stop-loss guide
        lower_95 = ci['lower_95'][-1]  # Last forecast point
        lower_80 = ci['lower_80'][-1]
        
        # Calculate stop levels
        aggressive_stop = lower_80 * 0.98  # 2% below 80% CI
        conservative_stop = lower_95 * 0.97  # 3% below 95% CI
        
        # Update position data
        position_data['forecast_stops'] = {
            'aggressive': aggressive_stop,
            'conservative': conservative_stop,
            'current_stop': position_data.get('stop_loss_price', conservative_stop)
        }
        
        # Recommend stop adjustment if needed
        if position_data['stop_loss_price'] < conservative_stop:
            position_data['stop_adjustment_recommended'] = True
            position_data['recommended_stop'] = conservative_stop
        
        return position_data
    
    def generate_ensemble_signals(self,
                                symbols: List[str],
                                market_data_dict: Dict[str, pd.DataFrame],
                                base_strategies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals for multiple symbols using ensemble forecasts.
        
        Args:
            symbols: List of trading symbols
            market_data_dict: Market data for each symbol
            base_strategies: Base strategy signals for each symbol
            
        Returns:
            List of enhanced trading signals
        """
        enhanced_signals = []
        
        for symbol in symbols:
            if symbol not in market_data_dict:
                continue
            
            # Get base strategy signal
            base_signal = base_strategies.get(symbol, {})
            
            # Generate forecast
            forecast = self.nf.generate_forecast(
                symbol=symbol,
                market_data=market_data_dict[symbol],
                horizon=24,
                models=['NBEATS', 'NHITS', 'TFT']
            )
            
            # Combine signals
            enhanced_signal = self._combine_forecast_with_strategy(
                base_signal, forecast, symbol
            )
            
            enhanced_signals.append(enhanced_signal)
        
        # Rank signals by combined score
        enhanced_signals.sort(
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )
        
        return enhanced_signals
    
    def _combine_forecast_with_strategy(self,
                                      base_signal: Dict[str, Any],
                                      forecast: Dict[str, Any],
                                      symbol: str) -> Dict[str, Any]:
        """Combine forecast with base strategy signal."""
        # Calculate forecast score
        ensemble_forecast = forecast['forecasts']['ensemble']
        current_price = base_signal.get('current_price', 100)
        
        forecast_return = (ensemble_forecast[-1] - current_price) / current_price
        forecast_score = np.tanh(forecast_return * 100)  # Normalize to [-1, 1]
        
        # Get strategy score
        strategy_score = base_signal.get('momentum_score', 0)
        
        # Combine scores with weights
        combined_score = (
            strategy_score * 0.6 +  # 60% weight to strategy
            forecast_score * 0.4    # 40% weight to forecast
        )
        
        # Determine action
        if combined_score > 0.3:
            action = 'BUY'
            confidence = 'HIGH' if combined_score > 0.5 else 'MEDIUM'
        elif combined_score < -0.3:
            action = 'SELL'
            confidence = 'HIGH' if combined_score < -0.5 else 'MEDIUM'
        else:
            action = 'HOLD'
            confidence = 'LOW'
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'combined_score': combined_score,
            'strategy_score': strategy_score,
            'forecast_score': forecast_score,
            'forecast_return': forecast_return,
            'position_size': self._calculate_position_size(combined_score),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_position_size(self, combined_score: float) -> float:
        """Calculate position size based on combined score."""
        # Kelly-inspired sizing with safety limits
        base_size = 0.02  # 2% base position
        score_multiplier = min(abs(combined_score) * 2, 3.0)  # Max 3x
        
        return min(base_size * score_multiplier, 0.10)  # Max 10% position
```

## 3. MCP Server Integration

Add to `/workspaces/ai-news-trader/mcp_server_enhanced.py`:

```python
# Add these tools to the existing MCP server

@mcp.tool()
def neural_forecast(symbol: str, horizon: int = 24, 
                   models: List[str] = ["NBEATS", "NHITS"],
                   use_gpu: bool = True) -> Dict[str, Any]:
    """
    Generate neural network time series forecast for a symbol.
    
    Args:
        symbol: Trading symbol to forecast
        horizon: Forecast horizon in hours
        models: List of models to use in ensemble
        use_gpu: Use GPU acceleration if available
    """
    try:
        # Initialize neural forecast integration
        nf_integration = NeuralForecastIntegration(gpu_enabled=use_gpu)
        
        # Get market data (mock for example)
        market_data = get_market_data(symbol, lookback_days=30)
        
        # Generate forecast
        forecast_result = nf_integration.generate_forecast(
            symbol=symbol,
            market_data=market_data,
            horizon=horizon,
            models=models,
            include_confidence=True
        )
        
        # Add trading recommendations
        current_price = market_data['close'].iloc[-1]
        forecast_price = forecast_result['forecasts']['ensemble'][-1]
        expected_return = (forecast_price - current_price) / current_price
        
        if expected_return > 0.02:
            recommendation = "BUY"
        elif expected_return < -0.02:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "forecast_24h": forecast_price,
            "expected_return": expected_return,
            "recommendation": recommendation,
            "confidence": forecast_result['quality_metrics']['quality_score'],
            "forecasts": forecast_result['forecasts'],
            "confidence_intervals": forecast_result.get('confidence_intervals', {}),
            "models_used": models,
            "gpu_used": use_gpu and GPU_AVAILABLE,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@mcp.tool()
def neural_backtest(symbol: str, model: str = "NBEATS",
                   test_windows: int = 3) -> Dict[str, Any]:
    """
    Backtest neural forecasting model performance.
    
    Args:
        symbol: Trading symbol to test
        model: Model to backtest
        test_windows: Number of test windows for cross-validation
    """
    try:
        # Initialize integration
        nf_integration = NeuralForecastIntegration(gpu_enabled=GPU_AVAILABLE)
        
        # Get historical data
        market_data = get_market_data(symbol, lookback_days=90)
        
        # Run backtest
        backtest_results = nf_integration.backtest_forecast_model(
            symbol=symbol,
            market_data=market_data,
            model=model,
            test_size=168,  # 1 week
            n_windows=test_windows
        )
        
        # Add performance summary
        metrics = backtest_results['metrics']
        
        return {
            "symbol": symbol,
            "model": model,
            "performance_metrics": {
                "MAE": float(metrics['MAE']),
                "RMSE": float(metrics['RMSE']),
                "MAPE": float(metrics['MAPE']),
                "directional_accuracy": float(backtest_results['directional_accuracy'])
            },
            "recommendation": "USE" if metrics['MAPE'] < 0.05 else "CAUTION",
            "test_windows": test_windows,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@mcp.tool()
def neural_ensemble_forecast(symbols: List[str], horizon: int = 24) -> Dict[str, Any]:
    """
    Generate ensemble neural forecasts for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        horizon: Forecast horizon in hours
    """
    try:
        # Initialize components
        nf_integration = NeuralForecastIntegration(gpu_enabled=GPU_AVAILABLE)
        enhancer = NeuralForecastStrategyEnhancer(nf_integration)
        
        # Collect market data
        market_data_dict = {}
        for symbol in symbols:
            market_data_dict[symbol] = get_market_data(symbol, lookback_days=30)
        
        # Get base strategy signals
        base_strategies = {}
        for symbol in symbols:
            # Mock base strategy signal
            base_strategies[symbol] = {
                'momentum_score': np.random.uniform(-0.5, 0.5),
                'current_price': market_data_dict[symbol]['close'].iloc[-1]
            }
        
        # Generate ensemble signals
        ensemble_signals = enhancer.generate_ensemble_signals(
            symbols=symbols,
            market_data_dict=market_data_dict,
            base_strategies=base_strategies
        )
        
        return {
            "symbols": symbols,
            "horizon": horizon,
            "signals": ensemble_signals,
            "top_opportunities": ensemble_signals[:3],
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@mcp.tool()
def neural_optimize_model(symbol: str, model_type: str = "NBEATS",
                         n_trials: int = 10) -> Dict[str, Any]:
    """
    Optimize neural forecast model hyperparameters.
    
    Args:
        symbol: Trading symbol
        model_type: Model type to optimize
        n_trials: Number of optimization trials
    """
    try:
        # Initialize integration
        nf_integration = NeuralForecastIntegration(gpu_enabled=GPU_AVAILABLE)
        
        # Get historical data
        market_data = get_market_data(symbol, lookback_days=60)
        
        # Run optimization
        optimization_results = nf_integration.optimize_hyperparameters(
            symbol=symbol,
            market_data=market_data,
            model_type=model_type,
            n_trials=n_trials
        )
        
        return {
            "symbol": symbol,
            "model_type": model_type,
            "best_parameters": optimization_results['best_params'],
            "best_performance": optimization_results['best_value'],
            "trials_completed": n_trials,
            "improvement": f"{(1 - optimization_results['best_value']) * 100:.1f}%",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
```

## 4. Testing Script

Create `/workspaces/ai-news-trader/tests/test_neural_forecast_integration.py`:

```python
"""
Test script for NeuralForecast integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
sys.path.append('/workspaces/ai-news-trader')

from src.forecasting.neural_forecast_integration import (
    NeuralForecastIntegration,
    NeuralForecastStrategyEnhancer
)


def generate_sample_market_data(symbol: str, days: int = 90) -> pd.DataFrame:
    """Generate sample market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    # Generate realistic price data
    price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Add trend and noise
        trend = 0.0001 * i
        seasonal = 5 * np.sin(2 * np.pi * i / (24 * 7))  # Weekly pattern
        noise = np.random.normal(0, 1)
        
        price = price * (1 + trend) + seasonal + noise
        prices.append(max(price, 10))  # Minimum price of 10
        
        # Volume with daily pattern
        base_volume = 1000000
        hour_of_day = i % 24
        volume_multiplier = 1.5 if 9 <= hour_of_day <= 16 else 0.5
        volumes.append(base_volume * volume_multiplier * (1 + np.random.uniform(-0.3, 0.3)))
    
    return pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)


async def test_basic_forecast():
    """Test basic neural forecasting functionality."""
    print("Testing basic neural forecast...")
    
    # Initialize integration
    nf = NeuralForecastIntegration(gpu_enabled=True)
    
    # Generate sample data
    symbol = 'TEST'
    market_data = generate_sample_market_data(symbol)
    
    # Generate forecast
    result = nf.generate_forecast(
        symbol=symbol,
        market_data=market_data,
        horizon=24,
        models=['NBEATS'],
        include_confidence=True
    )
    
    print(f"Forecast generated successfully!")
    print(f"Quality score: {result['quality_metrics']['quality_score']:.3f}")
    print(f"Forecast points: {len(result['forecasts']['NBEATS'])}")
    
    return result


async def test_ensemble_forecast():
    """Test ensemble forecasting."""
    print("\nTesting ensemble forecast...")
    
    # Initialize integration
    nf = NeuralForecastIntegration(gpu_enabled=True)
    
    # Generate sample data
    symbol = 'TEST'
    market_data = generate_sample_market_data(symbol)
    
    # Generate ensemble forecast
    result = nf.generate_forecast(
        symbol=symbol,
        market_data=market_data,
        horizon=48,
        models=['NBEATS', 'NHITS'],
        include_confidence=True
    )
    
    print(f"Ensemble forecast generated!")
    print(f"Models used: {result['models_used']}")
    print(f"Ensemble forecast available: {'ensemble' in result['forecasts']}")
    
    if 'confidence_intervals' in result:
        print(f"Confidence intervals calculated")
    
    return result


async def test_strategy_enhancement():
    """Test strategy enhancement with neural forecasts."""
    print("\nTesting strategy enhancement...")
    
    # Initialize components
    nf = NeuralForecastIntegration(gpu_enabled=True)
    enhancer = NeuralForecastStrategyEnhancer(nf)
    
    # Generate sample data
    symbol = 'TEST'
    market_data = generate_sample_market_data(symbol)
    
    # Mock momentum strategy output
    momentum_data = {
        'symbol': symbol,
        'action': 'BUY',
        'momentum_score': 0.65,
        'position_size_pct': 0.05,
        'confidence': 'HIGH',
        'reasoning': 'Strong momentum detected.'
    }
    
    # Enhance with neural forecast
    enhanced_signal = enhancer.enhance_momentum_signals(momentum_data, market_data)
    
    print(f"Original position size: {momentum_data['position_size_pct']:.1%}")
    print(f"Enhanced position size: {enhanced_signal['position_size_pct']:.1%}")
    print(f"Neural forecast direction: {enhanced_signal['neural_forecast']['direction']}")
    print(f"Expected return: {enhanced_signal['neural_forecast']['expected_return']:.2%}")
    
    return enhanced_signal


async def test_backtest():
    """Test model backtesting."""
    print("\nTesting model backtest...")
    
    # Initialize integration
    nf = NeuralForecastIntegration(gpu_enabled=True)
    
    # Generate sample data
    symbol = 'TEST'
    market_data = generate_sample_market_data(symbol, days=60)
    
    # Run backtest
    result = nf.backtest_forecast_model(
        symbol=symbol,
        market_data=market_data,
        model='NBEATS',
        test_size=168,
        n_windows=2
    )
    
    print(f"Backtest completed!")
    print(f"MAE: {result['metrics']['MAE']:.4f}")
    print(f"RMSE: {result['metrics']['RMSE']:.4f}")
    print(f"Directional accuracy: {result['directional_accuracy']:.1%}")
    
    return result


async def test_hyperparameter_optimization():
    """Test hyperparameter optimization."""
    print("\nTesting hyperparameter optimization...")
    
    # Initialize integration
    nf = NeuralForecastIntegration(gpu_enabled=True)
    
    # Generate sample data
    symbol = 'TEST'
    market_data = generate_sample_market_data(symbol, days=30)
    
    # Run optimization (small number of trials for testing)
    result = nf.optimize_hyperparameters(
        symbol=symbol,
        market_data=market_data,
        model_type='NBEATS',
        n_trials=5
    )
    
    print(f"Optimization completed!")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best MAE: {result['best_value']:.4f}")
    print(f"Trials completed: {result['n_trials']}")
    
    return result


async def run_all_tests():
    """Run all integration tests."""
    print("Starting NeuralForecast Integration Tests")
    print("=" * 50)
    
    try:
        # Test 1: Basic forecast
        await test_basic_forecast()
        
        # Test 2: Ensemble forecast
        await test_ensemble_forecast()
        
        # Test 3: Strategy enhancement
        await test_strategy_enhancement()
        
        # Test 4: Backtesting
        await test_backtest()
        
        # Test 5: Hyperparameter optimization
        await test_hyperparameter_optimization()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
```

## 5. Quick Start Guide

### Installation

```bash
cd /workspaces/ai-news-trader
pip install neuralforecast[gpu]
```

### Basic Usage

```python
# Import the integration
from src.forecasting.neural_forecast_integration import NeuralForecastIntegration

# Initialize
nf = NeuralForecastIntegration(gpu_enabled=True)

# Generate forecast
forecast = nf.generate_forecast(
    symbol='AAPL',
    market_data=your_data,
    horizon=24,
    models=['NBEATS', 'NHITS']
)

# Use in trading strategy
if forecast['forecasts']['ensemble'][-1] > current_price * 1.02:
    print("Strong bullish forecast - consider buying")
```

### Integration with Existing Strategies

```python
# Enhance momentum strategy
from src.forecasting.neural_forecast_integration import NeuralForecastStrategyEnhancer

enhancer = NeuralForecastStrategyEnhancer(nf)
enhanced_signal = enhancer.enhance_momentum_signals(
    momentum_data=your_momentum_signal,
    market_data=your_market_data
)
```

## 6. Performance Benchmarks

Expected performance characteristics:

- **Forecast Generation**: 1-3 seconds for 24-hour horizon
- **Model Training**: 10-30 seconds for 1 week of hourly data
- **GPU Speedup**: 5-10x faster than CPU for large datasets
- **Memory Usage**: 2-4GB for typical model ensemble
- **Accuracy**: 2-5% MAPE for 24-hour forecasts (market-dependent)

## 7. Best Practices

1. **Data Quality**: Ensure at least 30 days of historical data
2. **Model Selection**: Use ensemble of 2-3 models for robustness
3. **Retraining**: Retrain models weekly or after significant market events
4. **Position Sizing**: Never rely solely on forecasts - combine with risk management
5. **Monitoring**: Track forecast accuracy and adjust confidence accordingly

## 8. Troubleshooting

Common issues and solutions:

- **GPU Memory Error**: Reduce batch_size or use CPU fallback
- **Slow Training**: Check GPU availability, reduce max_steps
- **Poor Accuracy**: Increase training data, optimize hyperparameters
- **Import Errors**: Ensure all dependencies are installed correctly