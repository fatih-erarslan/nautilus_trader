# MCP Tool: neural_evaluate

## Overview
Evaluate a trained neural model on test data with specified metrics. This tool provides comprehensive performance assessment including accuracy metrics, error analysis, and directional accuracy for trading decisions.

## Tool Details
- **Name**: `mcp__ai-news-trader__neural_evaluate`
- **Category**: Neural Forecasting Tools
- **GPU Support**: Recommended for faster evaluation on large datasets
- **Evaluation Types**: Point forecasts, probabilistic forecasts, trading signals

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | *required* | ID of the trained model to evaluate |
| `test_data` | string | *required* | Path to test dataset (CSV, Parquet, or JSON) |
| `metrics` | array | `["mae", "rmse", "mape", "r2_score"]` | Metrics to calculate |
| `use_gpu` | boolean | `true` | Enable GPU acceleration for evaluation |

## Available Metrics
- **mae**: Mean Absolute Error
- **rmse**: Root Mean Square Error  
- **mape**: Mean Absolute Percentage Error
- **r2_score**: Coefficient of Determination
- **direction_accuracy**: Percentage of correct direction predictions
- **sharpe_ratio**: Risk-adjusted returns from trading signals
- **max_drawdown**: Maximum peak-to-trough decline
- **hit_rate**: Percentage of profitable trades
- **profit_factor**: Ratio of gross profits to gross losses
- **correlation**: Pearson correlation with actual values

## Return Value Structure
```json
{
  "model_id": "nhits_aapl_v3.2",
  "evaluation_results": {
    "mae": 1.245,
    "rmse": 1.876,
    "mape": 0.0085,
    "r2_score": 0.924,
    "direction_accuracy": 0.725,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.0523,
    "hit_rate": 0.682,
    "profit_factor": 2.15,
    "correlation": 0.962
  },
  "error_analysis": {
    "mean_error": 0.125,
    "std_error": 1.868,
    "skewness": 0.234,
    "kurtosis": 3.456,
    "error_percentiles": {
      "p5": -2.234,
      "p25": -0.876,
      "p50": 0.098,
      "p75": 1.123,
      "p95": 2.567
    }
  },
  "forecast_horizons": {
    "1h": {"mae": 0.523, "mape": 0.0032},
    "6h": {"mae": 0.987, "mape": 0.0061},
    "24h": {"mae": 1.245, "mape": 0.0085},
    "72h": {"mae": 2.134, "mape": 0.0142}
  },
  "trading_performance": {
    "total_trades": 245,
    "winning_trades": 167,
    "losing_trades": 78,
    "avg_win": 1.85,
    "avg_loss": -0.92,
    "win_rate": 0.682,
    "expectancy": 0.534,
    "annual_return": 0.285,
    "annual_volatility": 0.154
  },
  "metadata": {
    "test_samples": 25420,
    "test_period": "2024-01-01 to 2024-12-27",
    "evaluation_time": 4.2,
    "model_type": "NHITS",
    "features_used": ["price", "volume", "rsi", "sentiment"]
  }
}
```

## Examples

### Example 1: Standard Model Evaluation
```bash
# Evaluate model with default metrics
claude --mcp ai-news-trader "Evaluate my trained AAPL model on test data"

# The tool will be called as:
mcp__ai-news-trader__neural_evaluate({
  "model_id": "nhits_aapl_v3.2",
  "test_data": "/data/aapl_test_2024.csv",
  "metrics": ["mae", "rmse", "mape", "r2_score"],
  "use_gpu": true
})
```

### Example 2: Trading-Focused Evaluation
```bash
# Evaluate model for trading performance
claude --mcp ai-news-trader "Test my model's trading signal accuracy"

# The tool will be called as:
mcp__ai-news-trader__neural_evaluate({
  "model_id": "nbeats_spy_enhanced",
  "test_data": "/data/spy_test_with_signals.csv",
  "metrics": ["direction_accuracy", "sharpe_ratio", "hit_rate", "profit_factor"],
  "use_gpu": true
})
```

### Example 3: Comprehensive Evaluation
```bash
# Full evaluation with all metrics
claude --mcp ai-news-trader "Run complete evaluation of my portfolio model"

# The tool will be called as:
mcp__ai-news-trader__neural_evaluate({
  "model_id": "tft_portfolio_v2",
  "test_data": "/data/portfolio_test_2024.parquet",
  "metrics": ["mae", "rmse", "mape", "r2_score", "direction_accuracy", 
             "sharpe_ratio", "max_drawdown", "hit_rate", "profit_factor", "correlation"],
  "use_gpu": true
})
```

### Example 4: Multi-Period Evaluation
```bash
# Evaluate model across different market conditions
claude --mcp ai-news-trader "Test model performance on bull and bear market periods"

# Multiple evaluations will be run:
# Bull market period
mcp__ai-news-trader__neural_evaluate({
  "model_id": "nhits_market_adaptive",
  "test_data": "/data/bull_market_2023.csv",
  "metrics": ["mae", "direction_accuracy", "sharpe_ratio"],
  "use_gpu": true
})

# Bear market period
mcp__ai-news-trader__neural_evaluate({
  "model_id": "nhits_market_adaptive",
  "test_data": "/data/bear_market_2022.csv",
  "metrics": ["mae", "direction_accuracy", "sharpe_ratio"],
  "use_gpu": true
})
```

### Example 5: Error Analysis Focus
```bash
# Deep dive into prediction errors
claude --mcp ai-news-trader "Analyze prediction errors of my TSLA model"

# The tool will be called as:
mcp__ai-news-trader__neural_evaluate({
  "model_id": "deepar_tsla_v4",
  "test_data": "/data/tsla_test_detailed.json",
  "metrics": ["mae", "rmse", "mape", "correlation"],
  "use_gpu": true
})
# Claude will focus on the error_analysis section
```

## GPU Acceleration Notes

### Evaluation Performance
| Dataset Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 10K samples | 2.5s | 0.3s | 8.3x |
| 100K samples | 25s | 1.2s | 20.8x |
| 1M samples | 4 min | 8s | 30x |
| 10M samples | 45 min | 1.5 min | 30x |

### GPU Benefits
- Parallel metric computation
- Batch prediction processing
- Fast matrix operations for trading metrics
- Concurrent multi-horizon evaluation

## Evaluation Best Practices

### Test Data Requirements
1. **Temporal Split**: Never use future data in test set
2. **Market Conditions**: Include various market regimes
3. **Size**: Minimum 20% of training data size
4. **Features**: Must match training feature set exactly
5. **Frequency**: Same as training data (1min, 5min, etc.)

### Metric Selection Guide
| Use Case | Primary Metrics | Secondary Metrics |
|----------|----------------|-------------------|
| Price Forecasting | MAE, MAPE | RMSE, R² |
| Trading Signals | Direction Accuracy, Sharpe | Hit Rate, Profit Factor |
| Risk Management | Max Drawdown, VaR | Correlation, Volatility |
| Model Comparison | MAPE, Sharpe Ratio | All metrics |

### Interpretation Guidelines

#### Accuracy Thresholds
- **MAPE < 1%**: Excellent for short-term trading
- **MAPE 1-2%**: Good for swing trading
- **MAPE 2-5%**: Acceptable for long-term investing
- **Direction Accuracy > 60%**: Profitable trading possible
- **Sharpe Ratio > 1.5**: Strong risk-adjusted returns

#### Red Flags
- R² < 0.5: Model not capturing patterns
- Direction Accuracy < 55%: No better than random
- High RMSE with low MAE: Outlier problems
- Sharpe < 0.5: Poor risk-adjusted performance

## Advanced Evaluation Patterns

### Rolling Window Evaluation
```python
# Evaluate model performance over time
for month in test_months:
    results = neural_evaluate(
        model_id="model_v1",
        test_data=f"/data/{month}_test.csv"
    )
    track_performance_degradation(results)
```

### A/B Testing Models
```python
# Compare two models on same data
results_a = neural_evaluate("model_a", test_data)
results_b = neural_evaluate("model_b", test_data)
if results_b["sharpe_ratio"] > results_a["sharpe_ratio"] * 1.1:
    deploy_model("model_b")
```

### Ensemble Evaluation
```python
# Evaluate ensemble of models
models = ["nhits_v1", "nbeats_v1", "deepar_v1"]
ensemble_predictions = []
for model in models:
    results = neural_evaluate(model, test_data)
    ensemble_predictions.append(results)
# Combine and evaluate ensemble performance
```

## Performance Benchmarks

### Evaluation Speed by Model Type
| Model Type | 100K Samples (GPU) | 1M Samples (GPU) |
|------------|-------------------|------------------|
| NHITS | 0.8s | 6s |
| NBEATSx | 1.2s | 9s |
| DeepAR | 1.5s | 12s |
| TFT | 2.0s | 15s |

### Typical Performance Ranges
| Asset Class | Expected MAPE | Direction Accuracy | Sharpe Ratio |
|-------------|---------------|-------------------|--------------|
| Large Cap Stocks | 0.8-1.5% | 65-75% | 1.5-2.5 |
| Small Cap Stocks | 1.5-3.0% | 60-70% | 1.0-2.0 |
| Forex Majors | 0.3-0.8% | 70-80% | 2.0-3.0 |
| Crypto | 2.0-5.0% | 55-65% | 0.8-1.5 |

## Troubleshooting

### Issue: "Model not found"
- Check model_id with `neural_model_status`
- Ensure model training completed successfully
- Verify model file exists in `/models/` directory

### Issue: "Feature mismatch"
- Test data must have same features as training
- Check column names and order
- Verify data preprocessing is consistent

### Issue: "Poor performance metrics"
- Check for data drift between train and test
- Verify test period doesn't overlap training
- Consider retraining with recent data
- Use `neural_optimize` for better parameters

## Related Tools
- `neural_train`: Train models before evaluation
- `neural_backtest`: Full trading simulation
- `neural_optimize`: Improve model parameters
- `neural_model_status`: Check model details
- `performance_report`: Detailed analytics