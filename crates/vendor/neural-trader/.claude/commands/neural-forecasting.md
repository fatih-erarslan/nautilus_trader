# Neural Forecasting with Native MCP Tools

## Overview
AI-powered price prediction and model management using Claude Code's native MCP integration. All neural forecasting capabilities are available through natural language requests to Claude.

## Available MCP Neural Tools

### mcp__ai-news-trader__neural_forecast
Generate neural forecasts for trading symbols with advanced AI models.

**Natural Language Examples:**
```
"Generate a neural forecast for AAPL"
"Predict AAPL prices for the next 24 hours using GPU"
"Create a 48-hour forecast for TSLA with 99% confidence"
"Generate forecasts for AAPL, MSFT, and GOOGL"
```

**Direct MCP Parameters:**
- `symbol` (string): Trading symbol(s) to forecast
- `horizon` (int): Forecast horizon in hours (default: 24)
- `model_id` (string, optional): Specific model to use
- `confidence_level` (float): Confidence level (default: 0.95)
- `use_gpu` (bool): Enable GPU acceleration

**Example Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_forecast",
  "parameters": {
    "symbol": "AAPL",
    "horizon": 24,
    "confidence_level": 0.95,
    "use_gpu": true
  }
}
```

**Expected Return:**
```json
{
  "symbol": "AAPL",
  "current_price": 185.50,
  "forecast": {
    "24h_price": 187.25,
    "confidence_interval": [185.75, 188.75],
    "trend": "bullish",
    "volatility": "moderate",
    "model_confidence": 0.92
  },
  "model_used": "NHITS",
  "gpu_time_ms": 8.3
}
```

### mcp__ai-news-trader__neural_train
Train custom neural forecasting models on your trading data.

**Natural Language Examples:**
```
"Train a neural model on my trading_data.csv file"
"Train an NHITS model with 200 epochs using GPU"
"Create a Transformer model with custom learning rate 0.001"
"Train a model on large_dataset.csv with early stopping"
```

**Direct MCP Parameters:**
- `data_path` (string): Path to training dataset
- `model_type` (string): Model type [LSTM|Transformer|GRU|CNN_LSTM]
- `epochs` (int): Training epochs (default: 100)
- `batch_size` (int): Batch size (default: 32)
- `learning_rate` (float): Learning rate (default: 0.001)
- `validation_split` (float): Validation split (default: 0.2)
- `use_gpu` (bool): Enable GPU acceleration

**Example Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_train",
  "parameters": {
    "data_path": "trading_data.csv",
    "model_type": "Transformer",
    "epochs": 200,
    "use_gpu": true
  }
}
```

**Expected Return:**
```json
{
  "model_id": "transformer_20241215_001",
  "training_metrics": {
    "final_loss": 0.0012,
    "validation_loss": 0.0015,
    "epochs_completed": 200,
    "early_stopped": false
  },
  "performance": {
    "mae": 0.018,
    "rmse": 0.024,
    "training_time_seconds": 156.7,
    "gpu_utilized": true
  }
}
```

### mcp__ai-news-trader__neural_evaluate
Evaluate trained neural models on test data with comprehensive metrics.

**Natural Language Examples:**
```
"Evaluate my trained model on test_set.csv"
"Test the transformer_model_001 performance on new data"
"Check model accuracy with all available metrics"
"Evaluate the model and show detailed results"
```

**Direct MCP Parameters:**
- `model_id` (string): Model identifier to evaluate
- `test_data` (string): Path to test dataset
- `metrics` (list): Metrics to calculate ["mae", "rmse", "mape", "r2_score"]
- `use_gpu` (bool): Enable GPU acceleration

**Example Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_evaluate",
  "parameters": {
    "model_id": "transformer_20241215_001",
    "test_data": "test_set.csv",
    "metrics": ["mae", "rmse", "mape", "r2_score"],
    "use_gpu": true
  }
}
```

### mcp__ai-news-trader__neural_backtest
Run historical backtesting with neural forecasting models.

**Natural Language Examples:**
```
"Backtest my neural model on AAPL for 2024"
"Test the model's performance on TSLA from Jan to June 2024"
"Run historical validation with the transformer model"
"Backtest with $100k starting capital"
```

**Direct MCP Parameters:**
- `model_id` (string): Model identifier
- `start_date` (string): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)
- `benchmark` (string): Benchmark for comparison (default: "sp500")
- `rebalance_frequency` (string): Rebalancing frequency
- `use_gpu` (bool): Enable GPU acceleration

**Example Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_backtest",
  "parameters": {
    "model_id": "transformer_20241215_001",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "benchmark": "sp500",
    "use_gpu": true
  }
}
```

### mcp__ai-news-trader__neural_optimize
Optimize neural model hyperparameters for maximum performance.

**Natural Language Examples:**
```
"Optimize my neural model hyperparameters"
"Run 100 optimization trials to minimize MAE"
"Optimize the transformer model for best Sharpe ratio"
"Quick optimization with 20 trials using GPU"
```

**Direct MCP Parameters:**
- `model_id` (string): Model identifier
- `parameter_ranges` (dict): Parameter ranges for optimization
- `trials` (int): Number of optimization trials (default: 100)
- `optimization_metric` (string): Metric to optimize (default: "mae")
- `use_gpu` (bool): Enable GPU acceleration

**Example Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_optimize",
  "parameters": {
    "model_id": "transformer_20241215_001",
    "parameter_ranges": {
      "learning_rate": [0.0001, 0.01],
      "batch_size": [16, 128],
      "hidden_size": [64, 256]
    },
    "trials": 100,
    "optimization_metric": "mae",
    "use_gpu": true
  }
}
```

### mcp__ai-news-trader__neural_model_status
Get status and information about neural models.

**Natural Language Examples:**
```
"Show me the status of all my neural models"
"Check if the transformer model is healthy"
"Get performance metrics for model transformer_20241215_001"
"List all available neural models"
```

**Direct MCP Parameters:**
- `model_id` (string, optional): Specific model ID (returns all if not provided)

**Example Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_model_status",
  "parameters": {
    "model_id": "transformer_20241215_001"
  }
}
```

**Expected Return:**
```json
{
  "model_id": "transformer_20241215_001",
  "status": "active",
  "health": "healthy",
  "metrics": {
    "mae": 0.018,
    "last_updated": "2024-12-15T10:30:00Z",
    "predictions_made": 1547,
    "average_inference_time_ms": 8.3
  },
  "configuration": {
    "model_type": "Transformer",
    "gpu_enabled": true,
    "horizon": 24
  }
}
```

## Practical Neural Forecasting Workflows

### Daily Trading Forecast
```
"Good morning! Please:
1. Generate 24-hour forecasts for AAPL, MSFT, and GOOGL
2. Use GPU for fastest results
3. Show confidence intervals
4. Highlight any unusual predictions"
```

### Model Training Pipeline
```
"I have new trading data in 'latest_data.csv'. Please:
1. Train a Transformer model with 200 epochs
2. Use GPU acceleration
3. Evaluate on 20% validation split
4. Show me the training metrics"
```

### Model Performance Review
```
"Review my neural models:
1. Show status of all models
2. Evaluate the best model on recent data
3. Compare accuracy metrics
4. Suggest if retraining is needed"
```

### Advanced Forecast Analysis
```
"For my TSLA position:
1. Generate 48-hour forecast with 99% confidence
2. Compare with historical accuracy
3. Check if current volatility affects the forecast
4. Suggest position adjustments based on forecast"
```

## Performance Tips

### GPU Acceleration
- Claude automatically uses GPU when beneficial
- Explicitly request: "Use GPU for fastest results"
- 6,250x speedup for training and inference
- GPU memory automatically managed

### Batch Operations
```
"Generate forecasts for my entire watchlist using GPU:
AAPL, MSFT, GOOGL, TSLA, NVDA, META"
```

### Model Selection
```
"Which neural model is best for:
- Volatile crypto predictions?
- Stable blue-chip stocks?
- Intraday trading?"
```

## Integration with Other MCP Tools

### Combined Analysis
```
"For AAPL:
1. Analyze news sentiment (analyze_news)
2. If sentiment is positive, generate neural forecast
3. Run quick technical analysis
4. Suggest trading action"
```

### Risk-Adjusted Forecasting
```
"Generate neural forecasts for my portfolio and:
1. Calculate correlation between predictions
2. Assess forecast confidence vs current volatility
3. Suggest risk-adjusted position sizes"
```

## Best Practices for Neural MCP Tools

### Natural Language Tips
1. **Be specific about timeframes:**
   - "24-hour forecast" vs "next week forecast"
   - "Train on 2024 data" vs "recent data"

2. **Request confidence levels:**
   - "High confidence forecast (99%)"
   - "Show prediction intervals"

3. **Specify performance needs:**
   - "Use GPU for fastest training"
   - "Quick forecast, accuracy not critical"

### Model Management
```
# Regular maintenance request:
"Every Monday morning:
1. Check all model health status
2. Evaluate on last week's data
3. Retrain if accuracy drops below 90%
4. Alert me to any issues"
```

### Risk Management
```
# Forecast validation:
"Before using any forecast:
1. Check model's recent accuracy
2. Compare with actual vs predicted
3. Adjust confidence based on market volatility
4. Never trade solely on forecasts"
```

### Common Patterns
```
# Research workflow:
"Test different neural models:
1. Train LSTM, Transformer, and GRU on same data
2. Compare accuracy metrics
3. Select best for production
4. Document the results"

# Production monitoring:
"Daily model check:
1. Verify all models are healthy
2. Check prediction accuracy from yesterday
3. Flag any anomalies
4. Suggest retraining if needed"
```