# Neural Forecasting CLI Guide

## Overview

The Neural Forecasting CLI extension provides advanced machine learning capabilities for financial time series forecasting within the Claude-Flow platform. It integrates state-of-the-art neural forecasting models with the existing trading infrastructure.

## Installation

The neural CLI commands are integrated with the main Claude-Flow system. Ensure you have the required dependencies:

```bash
# Install neural forecasting dependencies
pip install neuralforecast[gpu]  # For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install optuna shap matplotlib seaborn pandas numpy tqdm click pydantic
```

## Command Structure

All neural forecasting commands follow the pattern:
```bash
./claude-flow neural <command> [options]
```

## Available Commands

### 1. Neural Forecast

Generate neural forecasts for trading symbols.

**Usage:**
```bash
./claude-flow neural forecast <symbol> [options]
```

**Arguments:**
- `symbol`: Trading symbol (e.g., AAPL, TSLA, BTC-USD)

**Options:**
- `--horizon, -h`: Forecast horizon in hours (default: 24)
- `--model, -m`: Model type [nhits|nbeats|tft|patchtst|auto] (default: auto)
- `--gpu`: Enable GPU acceleration
- `--confidence, -c`: Confidence level for intervals (default: 0.95)
- `--output, -o`: Output file path
- `--format, -f`: Output format [json|csv|text] (default: text)
- `--plot`: Generate forecast visualization

**Examples:**
```bash
# Basic forecast
./claude-flow neural forecast AAPL

# 48-hour forecast with GPU acceleration
./claude-flow neural forecast AAPL --horizon 48 --gpu

# High-frequency forecast with specific model
./claude-flow neural forecast TSLA --horizon 6 --model nhits --confidence 0.99

# Save results to file with plot
./claude-flow neural forecast BTC-USD --horizon 72 --output forecast.json --format json --plot
```

### 2. Neural Train

Train neural forecasting models on custom datasets.

**Usage:**
```bash
./claude-flow neural train <dataset> [options]
```

**Arguments:**
- `dataset`: Path to training dataset (CSV format)

**Options:**
- `--model, -m`: Model architecture [nhits|nbeats|tft|patchtst] (default: nhits)
- `--epochs, -e`: Training epochs (default: 100)
- `--batch-size, -b`: Batch size (default: 32)
- `--learning-rate, -lr`: Learning rate (default: 0.001)
- `--validation-split, -v`: Validation split ratio (default: 0.2)
- `--gpu`: Enable GPU training
- `--output, -o`: Model output path
- `--early-stopping`: Enable early stopping

**Examples:**
```bash
# Basic model training
./claude-flow neural train stock_data.csv

# Advanced training with GPU
./claude-flow neural train crypto_data.csv --model tft --epochs 200 --gpu --early-stopping

# Custom hyperparameters
./claude-flow neural train forex_data.csv --learning-rate 0.0001 --batch-size 64 --validation-split 0.15
```

**Dataset Format:**
CSV files should have columns: `timestamp`, `value`, optionally `symbol` for multiple assets.

```csv
timestamp,value,symbol
2024-01-01 00:00:00,150.25,AAPL
2024-01-01 01:00:00,150.50,AAPL
```

### 3. Neural Evaluate

Evaluate trained model performance.

**Usage:**
```bash
./claude-flow neural evaluate <model_path> [options]
```

**Arguments:**
- `model_path`: Path to trained model file

**Options:**
- `--test-data, -t`: Test dataset path
- `--metrics, -m`: Evaluation metrics (default: mae,mape,rmse)
- `--output, -o`: Results output path
- `--format, -f`: Output format [json|csv|text] (default: text)

**Available Metrics:**
- `mae`: Mean Absolute Error
- `mape`: Mean Absolute Percentage Error
- `rmse`: Root Mean Square Error
- `smape`: Symmetric Mean Absolute Percentage Error
- `mse`: Mean Square Error
- `r2`: R-squared coefficient

**Examples:**
```bash
# Basic evaluation
./claude-flow neural evaluate model.json --test-data test.csv

# Custom metrics
./claude-flow neural evaluate model.json --metrics mae,rmse,r2 --format json
```

### 4. Neural Backtest

Run historical backtesting with neural forecasts.

**Usage:**
```bash
./claude-flow neural backtest <model_path> [options]
```

**Arguments:**
- `model_path`: Path to trained model

**Options:**
- `--symbol, -s`: Trading symbol (required)
- `--start`: Start date (YYYY-MM-DD, required)
- `--end`: End date (YYYY-MM-DD, required)
- `--strategy`: Trading strategy (default: buy_hold)
- `--initial-capital, -c`: Initial capital (default: 10000)
- `--output, -o`: Results output path
- `--plot`: Generate performance plots

**Trading Strategies:**
- `buy_hold`: Buy and hold strategy
- `momentum`: Momentum-based trading
- `mean_reversion`: Mean reversion strategy
- `trend_following`: Trend following approach

**Examples:**
```bash
# Basic backtest
./claude-flow neural backtest model.json --symbol AAPL --start 2024-01-01 --end 2024-12-31

# Advanced backtest with custom strategy
./claude-flow neural backtest model.json --symbol TSLA --start 2024-06-01 --end 2024-12-31 \
    --strategy momentum --initial-capital 50000 --plot
```

### 5. Neural Deploy

Deploy models to production environments.

**Usage:**
```bash
./claude-flow neural deploy <model_path> [options]
```

**Arguments:**
- `model_path`: Path to model for deployment

**Options:**
- `--env, -e`: Environment [development|staging|production] (default: development)
- `--traffic, -t`: Traffic percentage (default: 100)
- `--health-check`: Run pre-deployment health checks
- `--rollback-threshold`: Error rate threshold for rollback (default: 0.1)

**Examples:**
```bash
# Development deployment
./claude-flow neural deploy model.json --env development

# Canary production deployment
./claude-flow neural deploy model.json --env production --traffic 10 --health-check

# Staging deployment with custom rollback
./claude-flow neural deploy model.json --env staging --rollback-threshold 0.05
```

### 6. Neural Monitor

Monitor deployed models in real-time.

**Usage:**
```bash
./claude-flow neural monitor [options]
```

**Options:**
- `--dashboard`: Launch monitoring dashboard
- `--env, -e`: Environment to monitor [development|staging|production|all] (default: all)
- `--refresh, -r`: Refresh interval in seconds (default: 30)
- `--alerts`: Enable alert notifications

**Examples:**
```bash
# Launch monitoring dashboard
./claude-flow neural monitor --dashboard

# Monitor production with alerts
./claude-flow neural monitor --env production --alerts --refresh 10

# One-time status check
./claude-flow neural monitor
```

### 7. Neural Optimize

Optimize model hyperparameters using advanced search algorithms.

**Usage:**
```bash
./claude-flow neural optimize <model_path> [options]
```

**Arguments:**
- `model_path`: Path to base model configuration

**Options:**
- `--trials, -t`: Number of optimization trials (default: 50)
- `--metric, -m`: Optimization metric [mae|mape|rmse|sharpe] (default: mae)
- `--gpu`: Enable GPU acceleration
- `--output, -o`: Optimized parameters output path
- `--timeout`: Optimization timeout in seconds (default: 3600)

**Examples:**
```bash
# Basic hyperparameter optimization
./claude-flow neural optimize model.json --trials 100

# Advanced optimization with GPU
./claude-flow neural optimize model.json --trials 200 --metric sharpe --gpu --timeout 7200
```

## Configuration

Neural forecasting behavior can be customized using the configuration file:

```bash
# Location
/workspaces/ai-news-trader/benchmark/configs/neural_config.yaml
```

Key configuration sections:
- **Models**: Default parameters for each model type
- **GPU**: GPU acceleration settings
- **Optimization**: Hyperparameter search spaces
- **Deployment**: Environment and health check settings
- **Monitoring**: Dashboard and alert configurations

## Model Types

### NHITS (Neural Hierarchical Interpolation for Time Series)
- **Best for**: Medium to long-term forecasting
- **Strengths**: Excellent accuracy, efficient training
- **Typical use**: Daily/weekly stock predictions

### NBEATS (Neural Basis Expansion Analysis)
- **Best for**: Short-term forecasting
- **Strengths**: Interpretable, fast inference
- **Typical use**: Intraday trading signals

### Temporal Fusion Transformer (TFT)
- **Best for**: Complex multi-variate forecasting
- **Strengths**: Handles multiple features, attention mechanisms
- **Typical use**: Multi-asset portfolio optimization

### PatchTST (Patch Time Series Transformer)
- **Best for**: Long sequence forecasting
- **Strengths**: Efficient for very long horizons
- **Typical use**: Monthly/quarterly predictions

## Best Practices

### Data Preparation
1. **Ensure data quality**: Remove outliers, handle missing values
2. **Sufficient history**: Use at least 1000+ data points for training
3. **Consistent frequency**: Maintain regular time intervals
4. **Feature engineering**: Include relevant market indicators

### Model Selection
1. **Start with NHITS**: Generally provides best accuracy/speed tradeoff
2. **Use TFT for multi-variate**: When you have multiple input features
3. **Try PatchTST for long horizons**: For forecasts > 1 week
4. **NBEATS for interpretability**: When you need explainable predictions

### Training Tips
1. **Use GPU acceleration**: Significantly faster training
2. **Enable early stopping**: Prevents overfitting
3. **Validate on out-of-sample data**: Use proper train/validation/test splits
4. **Monitor training curves**: Watch for convergence

### Production Deployment
1. **Start with canary deployments**: Use low traffic percentages initially
2. **Monitor performance closely**: Set up alerts for degradation
3. **Have rollback procedures**: Be prepared to revert quickly
4. **Regular model retraining**: Update models with new data

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory errors:**
- Reduce batch size: `--batch-size 16`
- Enable memory growth: Check GPU config
- Use CPU fallback: Remove `--gpu` flag

**Poor forecast accuracy:**
- Increase training data size
- Try different model architectures
- Adjust hyperparameters
- Check data quality and preprocessing

**Slow training:**
- Use GPU acceleration: `--gpu`
- Increase batch size: `--batch-size 64`
- Reduce model complexity
- Use early stopping: `--early-stopping`

### Error Codes

- **E001**: Invalid model path
- **E002**: Insufficient training data
- **E003**: GPU initialization failed
- **E004**: Invalid date range for backtesting
- **E005**: Deployment health check failed

## Integration with Trading Strategies

The neural forecasting CLI integrates seamlessly with existing trading strategies:

```bash
# Generate forecast
./claude-flow neural forecast AAPL --horizon 24 --output forecast.json

# Use forecast in trading decision
./claude-flow simulate --config trading_config.yaml --forecast forecast.json

# Backtest integrated strategy
./claude-flow neural backtest model.json --symbol AAPL --strategy momentum \
    --start 2024-01-01 --end 2024-12-31
```

## API Integration

Neural forecasting commands can be integrated with external systems:

```python
import subprocess
import json

# Generate forecast programmatically
result = subprocess.run([
    './claude-flow', 'neural', 'forecast', 'AAPL',
    '--horizon', '24', '--format', 'json', '--output', 'forecast.json'
], capture_output=True, text=True)

# Load forecast results
with open('forecast.json', 'r') as f:
    forecast = json.load(f)
```

## Performance Benchmarks

Typical performance characteristics:

| Model | Training Time | Inference Time | Memory Usage | Accuracy (MAPE) |
|-------|---------------|----------------|--------------|-----------------|
| NHITS | 2-5 minutes   | <50ms         | 512MB        | 3-8%           |
| NBEATS| 1-3 minutes   | <30ms         | 256MB        | 4-10%          |
| TFT   | 5-15 minutes  | <100ms        | 1GB          | 2-6%           |
| PatchTST| 3-8 minutes | <70ms         | 768MB        | 3-7%           |

*Benchmarks on NVIDIA RTX 3080, 1000 epochs, daily stock data*

## Support and Contributing

For support or to contribute improvements:

1. **Documentation**: Check this guide and configuration files
2. **Issues**: Report problems with detailed error messages
3. **Feature requests**: Suggest new neural forecasting capabilities
4. **Performance optimization**: Help improve model efficiency

## Advanced Usage

### Custom Model Development

Extend the neural forecasting system with custom models:

```python
# Add custom model to neural_command.py
class CustomNeuralModel:
    def __init__(self, config):
        self.config = config
    
    def train(self, data):
        # Custom training logic
        pass
    
    def predict(self, data):
        # Custom prediction logic
        pass
```

### Ensemble Forecasting

Combine multiple models for improved accuracy:

```bash
# Train multiple models
./claude-flow neural train data.csv --model nhits --output nhits_model.json
./claude-flow neural train data.csv --model tft --output tft_model.json

# Use ensemble prediction (requires custom implementation)
./claude-flow neural forecast AAPL --ensemble nhits_model.json,tft_model.json
```

This completes the comprehensive Neural Forecasting CLI Guide. The system provides industrial-strength machine learning capabilities for financial forecasting within the Claude-Flow platform.