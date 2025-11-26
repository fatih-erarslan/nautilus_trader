# Neural Forecasting CLI Examples

This document provides comprehensive examples of using the Claude-Flow Neural Forecasting CLI commands.

## Quick Start

### 1. Basic Stock Forecast

Generate a 24-hour forecast for Apple stock:

```bash
./claude-flow neural forecast AAPL
```

Output:
```
🔮 Generating neural forecast for AAPL
📊 Model: auto, Horizon: 24h, GPU: ✗
Forecasting ████████████████████████████████████████ 100%

📈 FORECAST RESULTS
==================================================
Symbol: AAPL
Model: auto
Forecast Range: 12.45
Trend: up
```

### 2. Advanced Forecast with GPU

Generate a 48-hour forecast with GPU acceleration and confidence intervals:

```bash
./claude-flow neural forecast TSLA --horizon 48 --gpu --confidence 0.99 --plot
```

This will:
- Use GPU acceleration for faster processing
- Generate 48-hour forecasts
- Provide 99% confidence intervals
- Save a forecast plot as PNG

### 3. Train a Custom Model

Train a neural forecasting model on your own data:

```bash
./claude-flow neural train my_stock_data.csv --model nhits --epochs 200 --gpu
```

Your CSV should have this format:
```csv
timestamp,value
2024-01-01 00:00:00,150.25
2024-01-01 01:00:00,150.50
2024-01-01 02:00:00,151.00
```

## Complete Workflow Examples

### Example 1: Cryptocurrency Trading Analysis

Complete workflow for Bitcoin analysis:

```bash
# Step 1: Generate forecast
./claude-flow neural forecast BTC-USD --horizon 72 --model tft --gpu \
    --output btc_forecast.json --format json

# Step 2: Train custom model on historical data
./claude-flow neural train crypto_historical.csv --model tft \
    --epochs 150 --batch-size 32 --gpu --output btc_model.json

# Step 3: Evaluate model performance
./claude-flow neural evaluate btc_model.json --test-data crypto_test.csv \
    --metrics mae,mape,rmse --format json --output evaluation.json

# Step 4: Run backtest
./claude-flow neural backtest btc_model.json --symbol BTC-USD \
    --start 2024-01-01 --end 2024-12-31 --initial-capital 50000 \
    --strategy momentum --plot --output backtest_results.json

# Step 5: Deploy to staging
./claude-flow neural deploy btc_model.json --env staging --traffic 50 --health-check

# Step 6: Monitor performance
./claude-flow neural monitor --dashboard --env staging --alerts
```

### Example 2: Multi-Asset Portfolio Optimization

Optimize forecasting across multiple assets:

```bash
# Train models for different assets
for symbol in AAPL GOOGL MSFT AMZN; do
    echo "Training model for $symbol"
    ./claude-flow neural train ${symbol}_data.csv \
        --model nhits --epochs 100 --gpu \
        --output ${symbol}_model.json
done

# Evaluate each model
for symbol in AAPL GOOGL MSFT AMZN; do
    ./claude-flow neural evaluate ${symbol}_model.json \
        --test-data ${symbol}_test.csv \
        --metrics mae,sharpe --format json \
        --output ${symbol}_evaluation.json
done

# Run portfolio backtest
./claude-flow neural backtest portfolio_model.json \
    --symbol "AAPL,GOOGL,MSFT,AMZN" --start 2024-01-01 --end 2024-12-31 \
    --strategy trend_following --initial-capital 100000 --plot
```

### Example 3: High-Frequency Trading Setup

Setup for high-frequency intraday trading:

```bash
# Generate short-term forecasts for multiple symbols
symbols=("AAPL" "TSLA" "NVDA" "META")
for symbol in "${symbols[@]}"; do
    ./claude-flow neural forecast $symbol --horizon 1 --model nbeats \
        --gpu --confidence 0.95 --format json \
        --output "${symbol}_1h_forecast.json" &
done
wait

# Train specialized short-term models
./claude-flow neural train intraday_data.csv --model nbeats \
    --epochs 300 --batch-size 128 --learning-rate 0.005 \
    --validation-split 0.1 --gpu --early-stopping \
    --output intraday_model.json

# Optimize hyperparameters
./claude-flow neural optimize intraday_model.json \
    --trials 100 --metric sharpe --gpu --timeout 3600 \
    --output optimized_params.json

# Deploy with canary rollout
./claude-flow neural deploy intraday_model.json \
    --env production --traffic 5 --health-check \
    --rollback-threshold 0.02
```

## Model-Specific Examples

### NHITS (Neural Hierarchical Interpolation)

Best for medium-term forecasting with excellent accuracy:

```bash
# Daily stock prediction
./claude-flow neural forecast AAPL --model nhits --horizon 24 --gpu

# Train for weekly forecasts
./claude-flow neural train weekly_data.csv --model nhits \
    --epochs 150 --batch-size 32 --learning-rate 0.001

# Optimize for longer horizons
./claude-flow neural optimize nhits_model.json \
    --trials 50 --metric mae --gpu
```

### NBEATS (Neural Basis Expansion Analysis)

Ideal for short-term, interpretable forecasts:

```bash
# Intraday trading signals
./claude-flow neural forecast TSLA --model nbeats --horizon 4 \
    --confidence 0.99 --plot

# Fast training for quick iterations
./claude-flow neural train hourly_data.csv --model nbeats \
    --epochs 100 --batch-size 64 --early-stopping

# Evaluate interpretability
./claude-flow neural evaluate nbeats_model.json \
    --test-data test.csv --metrics mae,mse,r2
```

### Temporal Fusion Transformer (TFT)

Perfect for complex multi-variate forecasting:

```bash
# Multi-feature forecasting
./claude-flow neural forecast AAPL --model tft --horizon 48 \
    --gpu --format json --output tft_forecast.json

# Train with multiple input features
./claude-flow neural train multivariate_data.csv --model tft \
    --epochs 200 --batch-size 16 --learning-rate 0.0003 --gpu

# Comprehensive backtesting
./claude-flow neural backtest tft_model.json --symbol AAPL \
    --start 2024-01-01 --end 2024-12-31 \
    --strategy trend_following --plot
```

### PatchTST (Patch Time Series Transformer)

Excellent for very long-term forecasting:

```bash
# Monthly predictions
./claude-flow neural forecast BTC-USD --model patchtst --horizon 720 \
    --gpu --confidence 0.95 --plot

# Long-term model training
./claude-flow neural train long_term_data.csv --model patchtst \
    --epochs 80 --batch-size 128 --learning-rate 0.005 --gpu

# Extended backtesting
./claude-flow neural backtest patchtst_model.json --symbol BTC-USD \
    --start 2023-01-01 --end 2024-12-31 \
    --strategy buy_hold --initial-capital 25000
```

## Production Deployment Examples

### Blue-Green Deployment

```bash
# Deploy to staging
./claude-flow neural deploy model_v2.json --env staging \
    --traffic 100 --health-check

# Canary production deployment
./claude-flow neural deploy model_v2.json --env production \
    --traffic 10 --health-check --rollback-threshold 0.05

# Monitor canary performance
./claude-flow neural monitor --env production --alerts --refresh 10

# Full production rollout
./claude-flow neural deploy model_v2.json --env production \
    --traffic 100 --health-check
```

### A/B Testing Setup

```bash
# Deploy model A
./claude-flow neural deploy model_a.json --env production \
    --traffic 50 --health-check

# Deploy model B
./claude-flow neural deploy model_b.json --env production \
    --traffic 50 --health-check

# Monitor both models
./claude-flow neural monitor --dashboard --env production --alerts
```

## Advanced Configuration Examples

### Custom Training Configuration

Create a custom training script:

```bash
#!/bin/bash
# advanced_training.sh

DATASET="$1"
MODEL_NAME="$2"

echo "Starting advanced training for $MODEL_NAME"

# Hyperparameter optimization first
./claude-flow neural optimize base_model.json \
    --trials 200 --metric mae --gpu --timeout 7200 \
    --output optimized_params.json

# Extract optimized parameters
LEARNING_RATE=$(jq -r '.best_parameters.learning_rate' optimized_params.json)
BATCH_SIZE=$(jq -r '.best_parameters.batch_size' optimized_params.json)
EPOCHS=$(jq -r '.best_parameters.epochs' optimized_params.json)

# Train with optimized parameters
./claude-flow neural train "$DATASET" --model "$MODEL_NAME" \
    --learning-rate "$LEARNING_RATE" --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" --gpu --early-stopping \
    --output "${MODEL_NAME}_optimized.json"

echo "Training completed: ${MODEL_NAME}_optimized.json"
```

Usage:
```bash
chmod +x advanced_training.sh
./advanced_training.sh stock_data.csv nhits
```

### Batch Processing Script

Process multiple symbols in parallel:

```bash
#!/bin/bash
# batch_forecast.sh

SYMBOLS=("AAPL" "GOOGL" "MSFT" "AMZN" "TSLA" "NVDA" "META" "NFLX")
HORIZON=24
OUTPUT_DIR="forecasts"

mkdir -p "$OUTPUT_DIR"

echo "Generating forecasts for ${#SYMBOLS[@]} symbols"

# Process in parallel
for symbol in "${SYMBOLS[@]}"; do
    {
        echo "Processing $symbol"
        ./claude-flow neural forecast "$symbol" \
            --horizon "$HORIZON" --gpu --model auto \
            --format json --output "$OUTPUT_DIR/${symbol}_forecast.json"
        echo "Completed $symbol"
    } &
done

# Wait for all processes to complete
wait

echo "All forecasts completed in $OUTPUT_DIR"

# Generate summary report
python3 -c "
import json
import os
from pathlib import Path

forecasts = []
for file in Path('$OUTPUT_DIR').glob('*_forecast.json'):
    with open(file) as f:
        data = json.load(f)
        forecasts.append({
            'symbol': data['symbol'],
            'trend': data['statistics']['trend_direction'],
            'mean_forecast': data['statistics']['mean_forecast']
        })

print('FORECAST SUMMARY')
print('=' * 50)
for f in forecasts:
    print(f\"{f['symbol']}: {f['trend']} (avg: {f['mean_forecast']:.2f})\")
"
```

### Monitoring Dashboard Script

Create a custom monitoring dashboard:

```bash
#!/bin/bash
# monitor_dashboard.sh

echo "🤖 Neural Model Monitoring Dashboard"
echo "=================================="

while true; do
    clear
    echo "Neural Model Status - $(date)"
    echo "=================================="
    
    # Get production status
    ./claude-flow neural monitor --env production | head -20
    
    echo ""
    echo "Recent Performance:"
    echo "------------------"
    
    # Check latest forecasts accuracy (mock)
    echo "Model Accuracy: 94.2%"
    echo "Latency: 45ms"
    echo "Uptime: 99.8%"
    echo "Requests/hour: 15,432"
    
    echo ""
    echo "Press Ctrl+C to exit, refreshing in 30s..."
    sleep 30
done
```

## Integration Examples

### With Existing Trading Systems

```python
# trading_integration.py
import subprocess
import json
from datetime import datetime

class NeuralForecastClient:
    def __init__(self):
        self.cli_path = "./claude-flow"
    
    def get_forecast(self, symbol, horizon=24, model="auto"):
        """Get neural forecast for a symbol."""
        cmd = [
            self.cli_path, "neural", "forecast", symbol,
            "--horizon", str(horizon),
            "--model", model,
            "--format", "json",
            "--gpu"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse forecast from stdout or output file
            return self._parse_forecast_output(result.stdout)
        else:
            raise Exception(f"Forecast failed: {result.stderr}")
    
    def _parse_forecast_output(self, output):
        """Parse forecast from CLI output."""
        # Implementation depends on actual output format
        return {"forecast": [150.0, 151.0, 152.0], "trend": "up"}

# Usage in trading system
client = NeuralForecastClient()
forecast = client.get_forecast("AAPL", horizon=48)
print(f"AAPL forecast: {forecast}")
```

### REST API Wrapper

```python
# api_wrapper.py
from flask import Flask, jsonify, request
import subprocess
import json

app = Flask(__name__)

@app.route('/forecast/<symbol>')
def forecast(symbol):
    horizon = request.args.get('horizon', 24, type=int)
    model = request.args.get('model', 'auto')
    
    try:
        cmd = [
            "./claude-flow", "neural", "forecast", symbol,
            "--horizon", str(horizon),
            "--model", model,
            "--format", "json",
            "--gpu"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({"status": "success", "forecast": "generated"})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Performance Optimization Examples

### GPU Memory Management

```bash
# Monitor GPU memory usage
nvidia-smi -l 1 &

# Train with memory-efficient settings
./claude-flow neural train large_dataset.csv --model nhits \
    --batch-size 16 --epochs 100 --gpu \
    --learning-rate 0.0001

# Optimize for memory usage
./claude-flow neural optimize model.json \
    --trials 25 --metric mae --gpu --timeout 1800
```

### Parallel Processing

```bash
# Train multiple models in parallel
./claude-flow neural train data1.csv --model nhits --gpu --output model1.json &
./claude-flow neural train data2.csv --model tft --gpu --output model2.json &
./claude-flow neural train data3.csv --model nbeats --gpu --output model3.json &
wait

echo "All models trained successfully"
```

These examples demonstrate the full capabilities of the Neural Forecasting CLI and show how to integrate it into production trading systems. Start with the basic examples and gradually work up to the more advanced use cases.