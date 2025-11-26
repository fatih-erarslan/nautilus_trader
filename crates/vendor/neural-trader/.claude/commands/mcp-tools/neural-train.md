# MCP Tool: neural_train

## Overview
Train a neural forecasting model with specified parameters and data. This tool enables custom model training on your own datasets, supporting both NHITS and NBEATSx architectures with full GPU acceleration.

## Tool Details
- **Name**: `mcp__ai-news-trader__neural_train`
- **Category**: Neural Forecasting Tools
- **GPU Support**: Highly recommended (6,250x speedup for large models)
- **Supported Models**: NHITS, NBEATSx, DeepAR, TFT (Temporal Fusion Transformer)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | string | *required* | Path to training data (CSV, Parquet, or JSON) |
| `model_type` | string | *required* | Model architecture: "nhits", "nbeats", "deepar", "tft" |
| `epochs` | integer | `100` | Number of training epochs |
| `batch_size` | integer | `32` | Training batch size |
| `learning_rate` | number | `0.001` | Initial learning rate |
| `validation_split` | number | `0.2` | Fraction of data for validation (0.1-0.4) |
| `use_gpu` | boolean | `true` | Enable GPU acceleration for training |

## Return Value Structure
```json
{
  "model_id": "nhits_custom_20241227_103045",
  "training_results": {
    "final_loss": 0.0125,
    "final_val_loss": 0.0142,
    "best_epoch": 87,
    "total_epochs": 100,
    "early_stopped": false,
    "convergence_rate": 0.92
  },
  "performance_metrics": {
    "train_mae": 0.0108,
    "train_rmse": 0.0145,
    "val_mae": 0.0132,
    "val_rmse": 0.0168,
    "val_mape": 0.85,
    "direction_accuracy": 0.725
  },
  "model_info": {
    "architecture": "NHITS",
    "parameters": 2548320,
    "input_features": ["open", "high", "low", "close", "volume", "rsi", "macd"],
    "forecast_horizon": 24,
    "lookback_window": 168,
    "saved_path": "/models/nhits_custom_20241227_103045.pkl"
  },
  "training_config": {
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "regularization": "dropout_0.1",
    "data_augmentation": true
  },
  "timing": {
    "total_time_seconds": 342.5,
    "time_per_epoch": 3.425,
    "gpu_utilization_avg": 0.89
  }
}
```

## Data Format Requirements

### CSV Format Example
```csv
timestamp,symbol,open,high,low,close,volume,rsi,macd,signal,sentiment
2024-01-01 09:30:00,AAPL,185.25,185.50,185.00,185.35,1250000,52.3,0.45,0.38,0.72
2024-01-01 09:31:00,AAPL,185.35,185.45,185.20,185.40,980000,52.5,0.46,0.39,0.71
```

### Required Columns
- `timestamp`: DateTime in ISO format
- `symbol`: Stock ticker
- `close`: Closing price (minimum requirement)

### Optional Feature Columns
- Price data: `open`, `high`, `low`
- Volume indicators: `volume`, `vwap`
- Technical indicators: `rsi`, `macd`, `bollinger_bands`
- Sentiment scores: `news_sentiment`, `social_sentiment`

## Examples

### Example 1: Basic Model Training
```bash
# Train NHITS model on Apple stock data
claude --mcp ai-news-trader "Train a neural model on my AAPL data at /data/aapl_2024.csv"

# The tool will be called as:
mcp__ai-news-trader__neural_train({
  "data_path": "/data/aapl_2024.csv",
  "model_type": "nhits",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "validation_split": 0.2,
  "use_gpu": true
})
```

### Example 2: Advanced NBEATSx Training
```bash
# Train NBEATSx with custom parameters
claude --mcp ai-news-trader "Train NBEATSx model on tech stocks with 200 epochs and batch size 64"

# The tool will be called as:
mcp__ai-news-trader__neural_train({
  "data_path": "/data/tech_stocks_combined.parquet",
  "model_type": "nbeats",
  "epochs": 200,
  "batch_size": 64,
  "learning_rate": 0.0005,
  "validation_split": 0.25,
  "use_gpu": true
})
```

### Example 3: Quick Prototype Training
```bash
# Fast training for prototyping
claude --mcp ai-news-trader "Quick train a model on SPY data with 50 epochs for testing"

# The tool will be called as:
mcp__ai-news-trader__neural_train({
  "data_path": "/data/spy_sample.csv",
  "model_type": "nhits",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.002,
  "validation_split": 0.15,
  "use_gpu": true
})
```

### Example 4: Production Model Training
```bash
# Train production-grade model with optimal settings
claude --mcp ai-news-trader "Train production model on full market data with DeepAR architecture"

# The tool will be called as:
mcp__ai-news-trader__neural_train({
  "data_path": "/data/market_full_2020_2024.parquet",
  "model_type": "deepar",
  "epochs": 300,
  "batch_size": 128,
  "learning_rate": 0.0001,
  "validation_split": 0.2,
  "use_gpu": true
})
```

### Example 5: Multi-Asset Model Training
```bash
# Train on portfolio of stocks
claude --mcp ai-news-trader "Train TFT model on my portfolio stocks for 30-day forecasting"

# The tool will be called as:
mcp__ai-news-trader__neural_train({
  "data_path": "/data/portfolio_combined.json",
  "model_type": "tft",
  "epochs": 150,
  "batch_size": 48,
  "learning_rate": 0.0003,
  "validation_split": 0.25,
  "use_gpu": true
})
```

## GPU Acceleration Notes

### Training Performance
| Dataset Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 10K samples | 45 min | 2 min | 22.5x |
| 100K samples | 8 hours | 15 min | 32x |
| 1M samples | 3 days | 35 min | 123x |
| 10M samples | 30 days | 4.8 hours | 150x |

### GPU Memory Requirements
- **NHITS**: ~2GB for standard configuration
- **NBEATSx**: ~3GB with external features
- **DeepAR**: ~4GB for probabilistic forecasting
- **TFT**: ~6GB for attention mechanisms

### Multi-GPU Support
- Automatic data parallelism for datasets > 1M samples
- Linear scaling up to 4 GPUs
- Distributed training for datasets > 10M samples

## Model Architecture Details

### NHITS
- **Best for**: Multi-horizon forecasting, seasonal patterns
- **Training time**: Fast convergence (50-100 epochs typical)
- **Interpretability**: High (hierarchical decomposition)

### NBEATSx
- **Best for**: Trend extraction, external variables
- **Training time**: Moderate (100-200 epochs)
- **Interpretability**: Medium (basis functions)

### DeepAR
- **Best for**: Probabilistic forecasting, uncertainty quantification
- **Training time**: Slow (200-300 epochs)
- **Interpretability**: Low (autoregressive RNN)

### TFT (Temporal Fusion Transformer)
- **Best for**: Complex patterns, multiple time series
- **Training time**: Very slow (300-500 epochs)
- **Interpretability**: Medium (attention weights)

## Performance Benchmarks

### Model Comparison (S&P 500 stocks)
| Model | Training Time (GPU) | Val MAPE | Parameters | Memory |
|-------|-------------------|----------|------------|---------|
| NHITS | 15 min | 0.82% | 2.5M | 2GB |
| NBEATSx | 25 min | 0.78% | 3.8M | 3GB |
| DeepAR | 45 min | 0.75% | 5.2M | 4GB |
| TFT | 90 min | 0.71% | 8.5M | 6GB |

## Best Practices

### Data Preparation
1. **Quality**: Clean data, handle missing values before training
2. **Features**: Include technical indicators and sentiment when available
3. **Normalization**: Data is auto-normalized, but pre-scaling helps
4. **Split**: Use 70/15/15 for train/val/test in production
5. **Frequency**: Ensure consistent time intervals (1min, 5min, 1hour, etc.)

### Training Strategy
1. **Start Simple**: Begin with NHITS, move to complex models if needed
2. **Early Stopping**: Built-in with patience=10 epochs
3. **Learning Rate**: Start with defaults, use neural_optimize for tuning
4. **Batch Size**: Larger = faster but may hurt generalization
5. **Validation**: Monitor val_loss to prevent overfitting

### Production Pipeline
```bash
# Step 1: Prepare data
./preprocess_data.py --input raw_data/ --output processed/

# Step 2: Train initial model
neural_train processed/train.csv nhits

# Step 3: Optimize hyperparameters
neural_optimize <model_id> --trials 100

# Step 4: Evaluate on test set
neural_evaluate <model_id> processed/test.csv

# Step 5: Deploy if metrics pass threshold
neural_deploy <model_id> --min-accuracy 0.7
```

## Common Issues and Solutions

### Issue: "GPU out of memory"
- Reduce batch_size (try 16 or 8)
- Use gradient accumulation
- Switch to smaller model (NHITS instead of TFT)

### Issue: "Validation loss increasing"
- Reduce learning_rate (try 0.0001)
- Increase dropout/regularization
- Check for data leakage

### Issue: "Training too slow"
- Ensure use_gpu=true
- Increase batch_size if GPU memory allows
- Use mixed precision training (automatic)

## Related Tools
- `neural_evaluate`: Test trained models
- `neural_optimize`: Hyperparameter tuning
- `neural_backtest`: Historical validation
- `neural_forecast`: Use trained models
- `neural_model_status`: Monitor training progress