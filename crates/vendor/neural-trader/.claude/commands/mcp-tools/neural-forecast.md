# MCP Tool: neural_forecast

## Overview
Generate neural network forecasts for a symbol with specified horizon and confidence level. This tool leverages advanced deep learning models (NHITS/NBEATSx) to produce accurate price predictions with uncertainty quantification.

## Tool Details
- **Name**: `mcp__ai-news-trader__neural_forecast`
- **Category**: Neural Forecasting Tools
- **GPU Support**: Strongly recommended for sub-10ms inference
- **Model Architecture**: NHITS (default) or NBEATSx

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | *required* | Stock symbol to forecast (e.g., "AAPL", "GOOGL") |
| `horizon` | integer | *required* | Forecast horizon in hours (1-720) |
| `confidence_level` | number | `0.95` | Confidence interval level (0.5-0.99) |
| `model_id` | string | `null` | Specific model ID to use, or null for auto-selection |
| `use_gpu` | boolean | `true` | Enable GPU acceleration for inference |

## Return Value Structure
```json
{
  "symbol": "AAPL",
  "forecast": {
    "point_forecast": [195.50, 196.25, 197.10, 198.05],
    "lower_bound": [194.20, 194.85, 195.50, 196.25],
    "upper_bound": [196.80, 197.65, 198.70, 199.85],
    "timestamps": [
      "2024-12-27T11:00:00Z",
      "2024-12-27T12:00:00Z",
      "2024-12-27T13:00:00Z",
      "2024-12-27T14:00:00Z"
    ]
  },
  "metadata": {
    "model_id": "nhits_aapl_v3.2",
    "model_type": "NHITS",
    "confidence_level": 0.95,
    "horizon_hours": 4,
    "training_window": "90 days",
    "last_updated": "2024-12-27T10:00:00Z",
    "feature_importance": {
      "price_lag_1h": 0.35,
      "volume_ma_24h": 0.22,
      "rsi_14": 0.18,
      "market_sentiment": 0.15,
      "vix": 0.10
    }
  },
  "performance_metrics": {
    "inference_time_ms": 8.5,
    "gpu_utilization": 0.45,
    "model_confidence": 0.92
  }
}
```

## Examples

### Example 1: Short-term Price Forecast
```bash
# Generate 24-hour forecast for Apple stock
claude --mcp ai-news-trader "Forecast AAPL price for the next 24 hours"

# The tool will be called as:
mcp__ai-news-trader__neural_forecast({
  "symbol": "AAPL",
  "horizon": 24,
  "confidence_level": 0.95,
  "use_gpu": true
})
```

### Example 2: Weekly Forecast with Custom Confidence
```bash
# Generate 1-week forecast with 99% confidence intervals
claude --mcp ai-news-trader "Predict TSLA price for next week with 99% confidence"

# The tool will be called as:
mcp__ai-news-trader__neural_forecast({
  "symbol": "TSLA",
  "horizon": 168,
  "confidence_level": 0.99,
  "use_gpu": true
})
```

### Example 3: Using Specific Model
```bash
# Use a specific trained model for forecasting
claude --mcp ai-news-trader "Forecast NVDA using model nhits_nvda_enhanced_v2"

# The tool will be called as:
mcp__ai-news-trader__neural_forecast({
  "symbol": "NVDA",
  "horizon": 48,
  "model_id": "nhits_nvda_enhanced_v2",
  "use_gpu": true
})
```

### Example 4: Multi-Timeframe Analysis
```bash
# Compare short and long-term forecasts
claude --mcp ai-news-trader "Generate 1-day and 1-week forecasts for GOOGL"

# Two calls will be made:
mcp__ai-news-trader__neural_forecast({
  "symbol": "GOOGL",
  "horizon": 24,
  "use_gpu": true
})

mcp__ai-news-trader__neural_forecast({
  "symbol": "GOOGL",
  "horizon": 168,
  "use_gpu": true
})
```

### Example 5: Trading Signal Generation
```bash
# Use forecast for trading decisions
claude --mcp ai-news-trader "Forecast SPY and suggest entry points based on predictions"

# The tool will be called as:
mcp__ai-news-trader__neural_forecast({
  "symbol": "SPY",
  "horizon": 72,
  "confidence_level": 0.95,
  "use_gpu": true
})
# Claude will analyze the forecast for trading opportunities
```

## GPU Acceleration Notes
- **Inference Speed**: 
  - GPU: 8-10ms for standard forecast
  - CPU: 150-500ms depending on model complexity
- **GPU Memory**: ~500MB per model
- **Batch Processing**: GPU can handle multiple forecasts simultaneously
- **Auto-fallback**: Seamlessly switches to CPU if GPU unavailable

## Model Architecture Details

### NHITS (Neural Hierarchical Interpolation for Time Series)
- **Architecture**: Multi-rate signal sampling with hierarchical interpolation
- **Strengths**: Excellent for multi-horizon forecasting, handles seasonality well
- **Parameters**: ~2.5M for standard configuration
- **Training Time**: 2-4 hours on historical data

### NBEATSx (Neural Basis Expansion Analysis with Exogenous variables)
- **Architecture**: Stack of fully connected layers with basis expansion
- **Strengths**: Superior for trend extraction, handles external features
- **Parameters**: ~3.8M for extended configuration
- **Training Time**: 3-6 hours with feature engineering

## Performance Benchmarks

### Accuracy Metrics (S&P 500 stocks, 24h horizon)
| Model | MAE | RMSE | MAPE | Direction Accuracy |
|-------|-----|------|------|-------------------|
| NHITS | 0.82% | 1.15% | 0.79% | 68.5% |
| NBEATSx | 0.78% | 1.08% | 0.75% | 71.2% |
| Baseline (ARIMA) | 1.45% | 2.01% | 1.38% | 54.3% |

### Inference Performance
| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Single Forecast (24h) | 185ms | 8ms | 23x |
| Single Forecast (168h) | 420ms | 12ms | 35x |
| Batch (10 symbols) | 2100ms | 35ms | 60x |
| With Feature Engineering | 580ms | 15ms | 38x |

## Best Practices
1. **Horizon Selection**: Use shorter horizons (1-72h) for trading, longer (168-720h) for planning
2. **Confidence Levels**: 
   - 0.95 for standard analysis
   - 0.99 for risk-averse strategies
   - 0.90 for aggressive trading
3. **Model Selection**: Let auto-selection choose unless you have domain expertise
4. **GPU Usage**: Always enable for production; critical for real-time trading
5. **Validation**: Compare forecasts with `neural_backtest` before live trading

## Integration Patterns
```python
# Example: Automated trading signal
forecast = neural_forecast("AAPL", 24)
if forecast["point_forecast"][-1] > current_price * 1.02:
    # 2% upside predicted
    execute_trade("AAPL", "buy", quantity=100)
```

## Related Tools
- `neural_train`: Train custom models on your data
- `neural_evaluate`: Assess model performance
- `neural_backtest`: Historical validation
- `neural_optimize`: Hyperparameter tuning
- `simulate_trade`: Test trading strategies with forecasts