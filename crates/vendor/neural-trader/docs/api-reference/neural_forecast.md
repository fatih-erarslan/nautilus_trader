# Neural Forecast API Reference

The Neural Forecast module provides advanced time-series forecasting capabilities using state-of-the-art deep learning models integrated with the AI News Trading Platform.

## Overview

The Neural Forecast integration leverages NHITS (Neural Hierarchical Interpolation for Time Series) and other neural architectures to provide:

- Sub-10ms inference latency
- 25% accuracy improvement over baseline models
- GPU acceleration with 6,250x speedup
- Real-time forecasting for trading decisions

## Core Classes

### NeuralForecastEngine

The main engine for neural forecasting operations.

```python
from src.forecasting.neural_forecast_integration import NeuralForecastEngine

engine = NeuralForecastEngine(
    model_type="nhits",
    gpu_acceleration=True,
    max_horizon=30,
    input_size=168
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | "nhits" | Model architecture: "nhits", "nbeats", "autoformer" |
| `gpu_acceleration` | bool | True | Enable GPU acceleration |
| `max_horizon` | int | 30 | Maximum forecast horizon in time steps |
| `input_size` | int | 168 | Input sequence length |
| `batch_size` | int | 32 | Batch size for inference |
| `device` | str | "cuda" | Device for computation: "cuda", "cpu" |

#### Methods

##### `fit(data, target_col, freq="D")`

Train the neural forecasting model.

**Parameters:**
- `data` (pd.DataFrame): Training data with datetime index
- `target_col` (str): Name of target column to forecast
- `freq` (str): Data frequency ("D", "H", "15min", etc.)

**Returns:**
- `NeuralForecastEngine`: Self for method chaining

**Example:**
```python
# Prepare training data
data = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=1000, freq='D'),
    'unique_id': 'AAPL',
    'y': np.random.randn(1000).cumsum()
})

# Train model
engine.fit(data, target_col='y', freq='D')
```

##### `predict(horizon=30, level=[80, 95])`

Generate forecasts with confidence intervals.

**Parameters:**
- `horizon` (int): Forecast horizon in time steps
- `level` (List[int]): Confidence levels for prediction intervals

**Returns:**
- `pd.DataFrame`: Forecasts with columns: ['ds', 'unique_id', 'NHITS', 'NHITS-lo-80', 'NHITS-hi-80', ...]

**Example:**
```python
# Generate 30-day forecast
forecasts = engine.predict(horizon=30, level=[80, 95])
print(forecasts.head())
```

##### `cross_validate(n_windows=3, step_size=7)`

Perform time series cross-validation.

**Parameters:**
- `n_windows` (int): Number of validation windows
- `step_size` (int): Step size between windows

**Returns:**
- `pd.DataFrame`: Cross-validation results with performance metrics

##### `optimize_hyperparameters(n_trials=100)`

Optimize model hyperparameters using Optuna.

**Parameters:**
- `n_trials` (int): Number of optimization trials

**Returns:**
- `Dict`: Best hyperparameters found

### NHITSModel

NHITS (Neural Hierarchical Interpolation for Time Series) model implementation.

```python
from src.forecasting.models.nhits import NHITSModel

model = NHITSModel(
    input_size=168,
    h=30,
    n_freq_downsample=[168, 24, 1],
    stack_types=['trend', 'seasonality'],
    n_blocks=[1, 1],
    mlp_units=[[512, 512], [512, 512]],
    interpolation_mode='linear',
    pooling_mode='MaxPool1d'
)
```

#### Key Features

- **Hierarchical Structure**: Multi-scale temporal modeling
- **Fast Inference**: Optimized for real-time predictions
- **GPU Acceleration**: CUDA-optimized operations
- **Memory Efficient**: Reduced memory footprint

## Trading Integration

### TradingForecastAdapter

Adapter class for integrating neural forecasts with trading strategies.

```python
from src.forecasting.trading_adapter import TradingForecastAdapter

adapter = TradingForecastAdapter(
    forecast_engine=engine,
    trading_strategy="momentum",
    risk_threshold=0.1
)
```

#### Methods

##### `generate_trading_signals(symbol, lookback=30)`

Generate trading signals based on forecasts.

**Parameters:**
- `symbol` (str): Trading symbol (e.g., "AAPL")
- `lookback` (int): Historical data lookback period

**Returns:**
- `Dict`: Trading signals with buy/sell recommendations

**Example:**
```python
signals = adapter.generate_trading_signals("AAPL", lookback=30)
print(f"Action: {signals['action']}, Confidence: {signals['confidence']}")
```

##### `backtest_strategy(start_date, end_date, symbols)`

Backtest trading strategy with neural forecasts.

**Parameters:**
- `start_date` (str): Backtest start date
- `end_date` (str): Backtest end date  
- `symbols` (List[str]): List of symbols to trade

**Returns:**
- `Dict`: Backtest results with performance metrics

## Performance Metrics

### Forecast Accuracy

| Metric | Description | Target |
|--------|-------------|--------|
| MAPE | Mean Absolute Percentage Error | < 5% |
| MAE | Mean Absolute Error | Minimized |
| RMSE | Root Mean Square Error | Minimized |
| MASE | Mean Absolute Scaled Error | < 1.0 |

### Runtime Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Single Forecast | < 10ms | 1000+ forecasts/sec |
| Batch Inference | < 50ms | 10,000+ forecasts/sec |
| Model Training | < 5 min | N/A |
| Hyperparameter Tuning | < 30 min | N/A |

## Error Handling

### Common Exceptions

```python
from src.forecasting.exceptions import (
    NeuralForecastError,
    ModelNotTrainedError,
    GPUNotAvailableError,
    InvalidDataError
)

# Example error handling
try:
    forecasts = engine.predict(horizon=30)
except ModelNotTrainedError:
    print("Model must be trained before prediction")
except GPUNotAvailableError:
    print("GPU acceleration not available, falling back to CPU")
except InvalidDataError as e:
    print(f"Invalid input data: {e}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_FORECAST_GPU` | Enable GPU acceleration | `true` |
| `NEURAL_FORECAST_DEVICE` | Computation device | `cuda` |
| `NEURAL_FORECAST_BATCH_SIZE` | Default batch size | `32` |
| `NEURAL_FORECAST_MAX_MEMORY` | Max GPU memory (GB) | `8` |

### Model Configuration

```python
config = {
    "model": {
        "type": "nhits",
        "input_size": 168,
        "horizon": 30,
        "n_freq_downsample": [168, 24, 1],
        "stack_types": ["trend", "seasonality"],
        "n_blocks": [1, 1],
        "mlp_units": [[512, 512], [512, 512]]
    },
    "training": {
        "max_epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32,
        "early_stopping_patience": 10
    },
    "gpu": {
        "enabled": True,
        "memory_fraction": 0.8,
        "allow_growth": True
    }
}
```

## Advanced Features

### Multi-Symbol Forecasting

```python
# Forecast multiple symbols simultaneously
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
multi_forecasts = engine.predict_multi(symbols, horizon=30)
```

### Probabilistic Forecasting

```python
# Generate probabilistic forecasts
prob_forecasts = engine.predict_probabilistic(
    horizon=30,
    num_samples=1000,
    level=[10, 20, 30, 40, 50, 60, 70, 80, 90]
)
```

### Real-time Streaming

```python
# Set up real-time forecast streaming
stream = engine.create_stream(
    symbols=["AAPL"],
    update_frequency="1min",
    forecast_horizon=30
)

for forecast in stream:
    print(f"New forecast: {forecast}")
```

## Integration Examples

### With MCP Server

```python
# Register neural forecast as MCP tool
from src.mcp.handlers.tools import register_tool

@register_tool("neural_forecast")
def neural_forecast_tool(symbol: str, horizon: int = 30):
    """Generate neural forecast for trading symbol."""
    engine = get_forecast_engine()
    forecast = engine.predict_symbol(symbol, horizon)
    return {
        "symbol": symbol,
        "forecast": forecast.to_dict(),
        "confidence": forecast.confidence_score
    }
```

### With Trading Strategies

```python
# Integrate with momentum strategy  
from src.trading.strategies.momentum_trader import MomentumEngine

class NeuralMomentumStrategy(MomentumEngine):
    def __init__(self, forecast_engine):
        super().__init__()
        self.forecast_engine = forecast_engine
    
    def generate_signals(self, symbol, data):
        # Get base momentum signals
        momentum_signals = super().generate_signals(symbol, data)
        
        # Get neural forecast
        forecast = self.forecast_engine.predict_symbol(symbol, horizon=5)
        
        # Combine signals
        combined_confidence = (
            momentum_signals['confidence'] * 0.7 + 
            forecast['confidence'] * 0.3
        )
        
        return {
            'action': momentum_signals['action'],
            'confidence': combined_confidence,
            'forecast': forecast
        }
```

## See Also

- [MCP Tools API Reference](mcp_tools.md)
- [CLI Reference](cli_reference.md)
- [Neural Forecast Tutorial](../tutorials/basic_forecasting.md)
- [GPU Optimization Guide](../tutorials/gpu_optimization.md)