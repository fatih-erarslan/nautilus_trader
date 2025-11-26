"""
Neural Forecast Integration for AI News Trading Platform.

This module provides NHITS (Neural Hierarchical Interpolation for Time Series) 
integration for enhanced financial forecasting in the AI News Trading Platform.

Components:
- NHITSForecaster: Core NHITS model implementation
- NeuralModelManager: Model lifecycle management
- StrategyEnhancer: Neural-enhanced trading strategies
- GPU acceleration support for fly.io deployment

Dependencies:
- neuralforecast>=1.6.4
- torch>=2.0.0 (for GPU support)
- numpy>=1.21.0
- pandas>=1.5.0
"""

from .nhits_forecaster import NHITSForecaster
from .neural_model_manager import NeuralModelManager
from .strategy_enhancer import StrategyEnhancer

__version__ = "1.0.0"
__author__ = "AI News Trading Platform"

__all__ = [
    "NHITSForecaster",
    "NeuralModelManager", 
    "StrategyEnhancer",
]