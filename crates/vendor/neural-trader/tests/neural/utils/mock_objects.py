"""
Mock Objects for Neural Forecasting Tests

Mock implementations of neural forecasting components for testing purposes.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from unittest.mock import Mock, MagicMock, AsyncMock
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MockNHITSConfig:
    """Mock NHITS configuration for testing."""
    h: int = 24
    input_size: int = 168
    n_freq_downsample: List[int] = field(default_factory=lambda: [4, 2, 1])
    n_pool_kernel_size: List[int] = field(default_factory=lambda: [4, 2, 1])
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 10
    early_stop_patience: int = 5
    use_gpu: bool = False
    mixed_precision: bool = False
    prediction_interval: int = 5
    confidence_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


class MockNHITSModel(nn.Module):
    """Mock NHITS model for testing."""
    
    def __init__(self, config: MockNHITSConfig):
        super().__init__()
        self.config = config
        
        # Simple linear layers to simulate NHITS
        self.encoder = nn.Linear(config.input_size, 256)
        self.decoder = nn.Linear(256, config.h)
        self.dropout = nn.Dropout(0.1)
        
        # Mock attributes
        self._is_fitted = False
        self._training_metrics = {}
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mock forward pass."""
        batch_size = x.shape[0]
        
        # Simple forward pass
        encoded = torch.relu(self.encoder(x))
        encoded = self.dropout(encoded)
        predictions = self.decoder(encoded)
        
        # Mock backcast (residual)
        backcast = torch.randn(batch_size, self.config.input_size)
        
        return {
            'point_forecast': predictions,
            'backcast_residual': backcast
        }
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
           validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, float]:
        """Mock training method."""
        self._is_fitted = True
        
        # Simulate training metrics
        self._training_metrics = {
            'train_loss': np.random.uniform(0.01, 0.1),
            'val_loss': np.random.uniform(0.02, 0.12),
            'epochs': np.random.randint(5, self.config.max_epochs),
            'training_time': np.random.uniform(10, 60)
        }
        
        return self._training_metrics
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Mock prediction method."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        with torch.no_grad():
            output = self.forward(X)
            return output['point_forecast']


class MockRealTimeEngine:
    """Mock real-time inference engine."""
    
    def __init__(self, model_path: str, config: MockNHITSConfig):
        self.model_path = model_path
        self.config = config
        self.model = MockNHITSModel(config)
        self.model._is_fitted = True
        
        # Performance metrics
        self.inference_times = []
        self.prediction_cache = {}
        
    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """Mock prediction with timing."""
        # Simulate inference time
        inference_time = np.random.uniform(5, 25)  # 5-25ms
        self.inference_times.append(inference_time)
        
        # Generate mock predictions
        predictions = np.random.randn(self.config.h) * 0.1
        
        return {
            'predictions': predictions,
            'inference_time_ms': inference_time,
            'timestamp': datetime.now(),
            'model_version': 'mock_v1.0',
            'confidence_intervals': {
                f'{int(level*100)}%': {
                    'lower': predictions - np.random.uniform(0.1, 0.5, len(predictions)),
                    'upper': predictions + np.random.uniform(0.1, 0.5, len(predictions))
                }
                for level in self.config.confidence_levels
            }
        }
    
    async def stream_predict(self, data_stream):
        """Mock streaming predictions."""
        async for data_point in data_stream:
            await asyncio.sleep(0.001)  # Simulate processing time
            
            if len(data_point) >= self.config.input_size:
                window_data = data_point[-self.config.input_size:]
                prediction = self.predict(window_data)
                yield prediction


class MockMultiAssetProcessor:
    """Mock multi-asset processor."""
    
    def __init__(self, assets: List[str], config: MockNHITSConfig):
        self.assets = assets
        self.config = config
        self.models = {asset: MockNHITSModel(config) for asset in assets}
        
        # Mark all models as fitted
        for model in self.models.values():
            model._is_fitted = True
    
    async def process_batch(self, asset_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Mock batch processing."""
        results = {}
        
        # Simulate parallel processing delay
        await asyncio.sleep(0.01 * len(asset_data))
        
        for asset, data in asset_data.items():
            if asset in self.models:
                predictions = np.random.randn(self.config.h) * 0.1
                
                results[asset] = {
                    'asset': asset,
                    'predictions': predictions,
                    'timestamp': datetime.now(),
                    'processing_time_ms': np.random.uniform(10, 50),
                    'confidence': np.random.uniform(0.7, 0.95)
                }
        
        return results


class MockEventAwareModel(MockNHITSModel):
    """Mock event-aware NHITS model."""
    
    def __init__(self, config: MockNHITSConfig, event_dim: int = 128):
        super().__init__(config)
        self.event_dim = event_dim
        
        # Add event processing components
        self.event_encoder = nn.Linear(event_dim, 64)
        self.event_impact = nn.Linear(64, config.h)
    
    def forward(self, x: torch.Tensor, events: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with event integration."""
        base_output = super().forward(x)
        
        if events is not None:
            # Process events
            event_features = torch.relu(self.event_encoder(events.mean(dim=1)))
            event_impact = self.event_impact(event_features)
            
            # Add event impact to predictions
            base_output['point_forecast'] += 0.1 * event_impact
            base_output['event_impact'] = event_impact
        
        return base_output


class MockMCPServer:
    """Mock MCP server for testing neural tools."""
    
    def __init__(self):
        self.tools = {}
        self.models = {}
        self.call_history = []
        
        # Register mock tools
        self._register_neural_tools()
    
    def _register_neural_tools(self):
        """Register neural forecasting tools."""
        self.tools = {
            'neural_forecast': self._neural_forecast,
            'neural_backtest': self._neural_backtest,
            'neural_train': self._neural_train,
            'neural_optimize': self._neural_optimize,
            'neural_analyze': self._neural_analyze,
            'neural_benchmark': self._neural_benchmark
        }
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool calling."""
        self.call_history.append({
            'tool': tool_name,
            'parameters': parameters,
            'timestamp': datetime.now()
        })
        
        if tool_name not in self.tools:
            return {
                'status': 'error',
                'error': f'Unknown tool: {tool_name}',
                'available_tools': list(self.tools.keys())
            }
        
        try:
            result = await self.tools[tool_name](parameters)
            return {
                'status': 'success',
                'result': result,
                'tool': tool_name,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'tool': tool_name
            }
    
    async def _neural_forecast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neural forecasting tool."""
        symbol = params.get('symbol', 'UNKNOWN')
        horizon = params.get('horizon', 24)
        confidence_level = params.get('confidence_level', 95)
        use_gpu = params.get('use_gpu', False)
        
        # Simulate processing time
        await asyncio.sleep(0.01 if use_gpu else 0.05)
        
        # Generate mock forecast
        forecast = np.random.randn(horizon) * 0.1
        
        return {
            'symbol': symbol,
            'forecast': forecast.tolist(),
            'horizon': horizon,
            'confidence_intervals': {
                f'{confidence_level}%': {
                    'lower': (forecast - 0.1).tolist(),
                    'upper': (forecast + 0.1).tolist()
                }
            },
            'metadata': {
                'model_version': 'mock_nhits_v1.0',
                'inference_time_ms': np.random.uniform(10, 50),
                'gpu_used': use_gpu,
                'accuracy_score': np.random.uniform(0.8, 0.95)
            }
        }
    
    async def _neural_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neural backtesting tool."""
        model_id = params.get('model_id', 'mock_model')
        start_date = params.get('start_date', '2023-01-01')
        end_date = params.get('end_date', '2023-12-31')
        metrics = params.get('metrics', ['mae', 'mape', 'sharpe'])
        
        await asyncio.sleep(0.1)  # Simulate backtest time
        
        # Generate mock backtest results
        results = {}
        for metric in metrics:
            if metric == 'mae':
                results[metric] = np.random.uniform(0.01, 0.1)
            elif metric == 'mape':
                results[metric] = np.random.uniform(1, 10)
            elif metric == 'sharpe':
                results[metric] = np.random.uniform(0.5, 2.5)
            elif metric == 'rmse':
                results[metric] = np.random.uniform(0.02, 0.15)
            else:
                results[metric] = np.random.uniform(0, 1)
        
        return {
            'model_id': model_id,
            'period': f'{start_date} to {end_date}',
            'metrics': results,
            'performance_summary': {
                'total_predictions': np.random.randint(1000, 10000),
                'accuracy_score': np.random.uniform(0.7, 0.95),
                'profit_factor': np.random.uniform(1.1, 2.5),
                'max_drawdown': np.random.uniform(0.05, 0.25)
            }
        }
    
    async def _neural_train(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neural training tool."""
        data_path = params.get('data_path', 'mock_data.csv')
        model_type = params.get('model_type', 'nhits')
        epochs = params.get('epochs', 100)
        use_gpu = params.get('use_gpu', False)
        
        # Simulate training time
        training_time = epochs * (0.1 if use_gpu else 0.5)
        await asyncio.sleep(min(training_time / 100, 2.0))  # Scale down for testing
        
        return {
            'model_id': f'trained_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'training_metrics': {
                'final_train_loss': np.random.uniform(0.01, 0.1),
                'final_val_loss': np.random.uniform(0.02, 0.12),
                'epochs_completed': epochs,
                'training_time_seconds': training_time,
                'best_epoch': np.random.randint(epochs//2, epochs)
            },
            'model_info': {
                'type': model_type,
                'parameters': np.random.randint(100000, 1000000),
                'input_size': params.get('input_size', 168),
                'horizon': params.get('horizon', 24),
                'gpu_used': use_gpu
            }
        }
    
    async def _neural_optimize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neural optimization tool."""
        model_id = params.get('model_id', 'mock_model')
        optimization_metric = params.get('optimization_metric', 'mae')
        max_trials = params.get('max_trials', 100)
        
        await asyncio.sleep(0.5)  # Simulate optimization time
        
        return {
            'model_id': model_id,
            'optimization_results': {
                'best_score': np.random.uniform(0.01, 0.05),
                'best_params': {
                    'learning_rate': np.random.uniform(1e-4, 1e-2),
                    'batch_size': np.random.choice([32, 64, 128, 256]),
                    'hidden_size': np.random.choice([128, 256, 512]),
                    'dropout_rate': np.random.uniform(0.1, 0.5)
                },
                'trials_completed': max_trials,
                'optimization_time_seconds': np.random.uniform(60, 300)
            },
            'performance_improvement': {
                'baseline_score': np.random.uniform(0.05, 0.1),
                'optimized_score': np.random.uniform(0.01, 0.05),
                'improvement_percent': np.random.uniform(20, 70)
            }
        }
    
    async def _neural_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neural analysis tool."""
        symbol = params.get('symbol', 'UNKNOWN')
        analysis_type = params.get('analysis_type', 'feature_importance')
        
        await asyncio.sleep(0.2)
        
        if analysis_type == 'feature_importance':
            features = ['price_lag_1', 'price_lag_24', 'volume', 'volatility', 'trend']
            importance_scores = np.random.dirichlet(np.ones(len(features)))
            
            return {
                'symbol': symbol,
                'analysis_type': analysis_type,
                'feature_importance': dict(zip(features, importance_scores)),
                'top_features': sorted(zip(features, importance_scores), 
                                     key=lambda x: x[1], reverse=True)[:3]
            }
        else:
            return {
                'symbol': symbol,
                'analysis_type': analysis_type,
                'results': f'Mock analysis results for {analysis_type}'
            }
    
    async def _neural_benchmark(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neural benchmarking tool."""
        model_id = params.get('model_id', 'mock_model')
        benchmark_type = params.get('benchmark_type', 'performance')
        
        await asyncio.sleep(0.3)
        
        return {
            'model_id': model_id,
            'benchmark_type': benchmark_type,
            'results': {
                'latency_ms': {
                    'mean': np.random.uniform(10, 50),
                    'p95': np.random.uniform(50, 100),
                    'p99': np.random.uniform(100, 200)
                },
                'throughput': {
                    'predictions_per_second': np.random.uniform(100, 1000),
                    'batches_per_second': np.random.uniform(10, 100)
                },
                'memory_usage': {
                    'gpu_memory_mb': np.random.uniform(100, 1000),
                    'cpu_memory_mb': np.random.uniform(50, 500)
                },
                'accuracy': {
                    'mae': np.random.uniform(0.01, 0.1),
                    'mape': np.random.uniform(1, 10),
                    'directional_accuracy': np.random.uniform(0.5, 0.8)
                }
            }
        }


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                 batch_size: int = 32, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        
        if isinstance(data, pd.DataFrame):
            self.total_samples = len(data)
        else:
            self.total_samples = min(len(df) for df in data.values())
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.total_samples:
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, self.total_samples)
        
        if isinstance(self.data, pd.DataFrame):
            batch = self.data.iloc[self.current_idx:end_idx]
        else:
            batch = {
                asset: df.iloc[self.current_idx:end_idx]
                for asset, df in self.data.items()
            }
        
        self.current_idx = end_idx
        return batch
    
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size


class MockModelRegistry:
    """Mock model registry for testing."""
    
    def __init__(self):
        self.models = {}
        self.metadata = {}
        self.versions = {}
    
    def register_model(self, model_id: str, model: Any, metadata: Dict[str, Any]) -> str:
        """Register a mock model."""
        version = f"v{len(self.versions.get(model_id, [])) + 1}.0"
        full_id = f"{model_id}_{version}"
        
        self.models[full_id] = model
        self.metadata[full_id] = {
            **metadata,
            'registered_at': datetime.now(),
            'version': version
        }
        
        if model_id not in self.versions:
            self.versions[model_id] = []
        self.versions[model_id].append(version)
        
        return full_id
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Get model by ID and version."""
        if version:
            full_id = f"{model_id}_{version}"
        else:
            # Get latest version
            if model_id not in self.versions:
                raise ValueError(f"Model {model_id} not found")
            latest_version = self.versions[model_id][-1]
            full_id = f"{model_id}_{latest_version}"
        
        if full_id not in self.models:
            raise ValueError(f"Model {full_id} not found")
        
        return self.models[full_id], self.metadata[full_id]
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions."""
        return self.versions.copy()


# Factory functions for creating mock objects
def create_mock_nhits_model(config: Optional[MockNHITSConfig] = None) -> MockNHITSModel:
    """Create a mock NHITS model."""
    if config is None:
        config = MockNHITSConfig()
    return MockNHITSModel(config)


def create_mock_mcp_server() -> MockMCPServer:
    """Create a mock MCP server."""
    return MockMCPServer()


def create_mock_real_time_engine(config: Optional[MockNHITSConfig] = None) -> MockRealTimeEngine:
    """Create a mock real-time engine."""
    if config is None:
        config = MockNHITSConfig()
    return MockRealTimeEngine("mock_model.pt", config)


def create_mock_multi_asset_processor(assets: List[str], 
                                    config: Optional[MockNHITSConfig] = None) -> MockMultiAssetProcessor:
    """Create a mock multi-asset processor."""
    if config is None:
        config = MockNHITSConfig()
    return MockMultiAssetProcessor(assets, config)


# Export all mock classes and functions
__all__ = [
    'MockNHITSConfig',
    'MockNHITSModel',
    'MockRealTimeEngine',
    'MockMultiAssetProcessor',
    'MockEventAwareModel',
    'MockMCPServer',
    'MockDataLoader',
    'MockModelRegistry',
    'create_mock_nhits_model',
    'create_mock_mcp_server',
    'create_mock_real_time_engine',
    'create_mock_multi_asset_processor'
]