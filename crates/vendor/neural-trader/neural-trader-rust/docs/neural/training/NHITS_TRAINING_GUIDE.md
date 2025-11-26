# NHITS Training Pipeline Guide

Complete guide for training NHITS (Neural Hierarchical Interpolation for Time Series) models in Neural Trader.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Training from Different Sources](#training-from-different-sources)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Model Checkpointing](#model-checkpointing)
7. [Validation & Metrics](#validation--metrics)
8. [GPU Acceleration](#gpu-acceleration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

NHITS is a state-of-the-art neural forecasting model that uses hierarchical interpolation for multi-horizon time series prediction. The model features:

- **Hierarchical architecture** with multiple stacks
- **Frequency downsampling** for multi-scale pattern capture
- **MLP blocks** for non-linear transformations
- **Backcast/forecast separation** for interpretability
- **Quantile forecasting** for probabilistic predictions

## Quick Start

### 1. Prepare Your Data

Your data should be in CSV or Parquet format with at least:
- A timestamp/date column
- A target column (e.g., "close" for stock prices)
- Optional: Additional features

Example CSV:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.0,101.0,99.0,100.5,1000000
2024-01-01 01:00:00,100.5,102.0,100.0,101.2,1200000
...
```

### 2. Train the Model

```bash
neural-trader train-neural \
  --data historical_data.csv \
  --target close \
  --input-size 168 \
  --horizon 24 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --output trained_model.safetensors
```

### 3. Monitor Training

Training progress shows:
```
Epoch 1/100: train_loss=2.453617, val_loss=2.387234, lr=1.00e-3, time=1.23s
Epoch 2/100: train_loss=1.892341, val_loss=1.945123, lr=1.00e-3, time=1.18s
...
Early stopping triggered after 45 epochs
```

## Configuration

### Basic Configuration

```bash
--model nhits              # Model type (nhits, lstm, transformer)
--data path/to/data.csv    # Training data path
--target close             # Target column name
--input-size 168           # Lookback window (e.g., 1 week hourly)
--horizon 24               # Forecast horizon (e.g., 24 hours)
--epochs 100               # Maximum training epochs
--batch-size 32            # Batch size
--lr 0.001                 # Learning rate
--output model.safetensors # Output path
```

### Advanced Configuration

```bash
# Model architecture
--hidden-size 512          # Hidden layer size (default: 512)
--dropout 0.1              # Dropout rate (default: 0.1)
--n-stacks 3               # Number of NHITS stacks (default: 3)

# Training settings
--optimizer adamw          # Optimizer: adam, adamw, sgd, rmsprop
--weight-decay 1e-5        # L2 regularization
--val-split 0.2            # Validation split ratio
--patience 10              # Early stopping patience

# Performance
--gpu 0                    # Use GPU device 0
--mixed-precision          # Enable FP16 training

# Checkpointing
--checkpoint-dir ./ckpts   # Save checkpoints here
--resume ./ckpts/best.pth  # Resume from checkpoint
```

## Training from Different Sources

### From CSV File

```rust
use nt_neural::{NHITSTrainer, NHITSTrainingConfig};

let mut trainer = NHITSTrainer::new(config)?;
let metrics = trainer.train_from_csv("data.csv", "close").await?;
```

### From Parquet (Faster for Large Datasets)

```rust
let metrics = trainer.train_from_parquet("data.parquet", "close").await?;
```

### From In-Memory DataFrame

```rust
use polars::prelude::*;

let df = df!(
    "timestamp" => timestamps,
    "close" => prices,
    "volume" => volumes
)?;

let metrics = trainer.train_from_dataframe(df, "close").await?;
```

## Hyperparameter Tuning

### Input Size Selection

**Rule of thumb**: Input size should capture relevant historical patterns.

```bash
# For hourly stock data:
--input-size 168   # 1 week (7 * 24)
--input-size 720   # 1 month (30 * 24)

# For daily data:
--input-size 30    # 1 month
--input-size 90    # 3 months
--input-size 252   # 1 trading year
```

### Horizon Selection

**Rule of thumb**: Start with shorter horizons and increase.

```bash
--horizon 1     # Next step prediction
--horizon 24    # 24 hours ahead
--horizon 168   # 1 week ahead
```

### Learning Rate

**Recommended ranges**:
- Adam/AdamW: 1e-4 to 1e-2
- SGD: 1e-3 to 1e-1

```bash
# Conservative (stable but slower)
--lr 0.0001

# Standard (good default)
--lr 0.001

# Aggressive (faster but may diverge)
--lr 0.01
```

### Batch Size

**Recommendations**:
- Small datasets (< 1000 samples): 8-16
- Medium datasets (1000-10000): 32-64
- Large datasets (> 10000): 64-128

```bash
--batch-size 32    # Good default
--batch-size 64    # Better for GPU utilization
```

### Hidden Size

**Recommendations**:
- Simple patterns: 128-256
- Complex patterns: 512-1024
- Very complex patterns: 1024-2048

```bash
--hidden-size 512  # Good default
```

### Number of Stacks

**Recommendations**:
- 2 stacks: Simple, faster training
- 3 stacks: Good default (captures short, medium, long patterns)
- 4+ stacks: Very complex patterns

```bash
--n-stacks 3       # Recommended default
```

## Model Checkpointing

### Automatic Checkpointing

```bash
neural-trader train-neural \
  --data data.csv \
  --checkpoint-dir ./checkpoints \
  --output model.safetensors
```

Creates:
```
checkpoints/
├── best_model.safetensors      # Best validation loss
├── checkpoint_metadata.json    # Metadata
├── checkpoint_epoch_10.safetensors
├── checkpoint_epoch_20.safetensors
└── ...
```

### Resume Training

```bash
neural-trader train-neural \
  --data data.csv \
  --resume ./checkpoints/best_model.safetensors \
  --output model_continued.safetensors
```

## Validation & Metrics

### Training Metrics

Tracked during training:
- **train_loss**: Training set loss (MSE)
- **val_loss**: Validation set loss
- **learning_rate**: Current learning rate
- **epoch_time**: Time per epoch

### Evaluation Metrics

After training:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **R²** (R-squared): Explained variance (0-1, higher is better)

### Target Metrics

**Good performance thresholds** (domain-dependent):
- Stock prices: MAPE < 5-10%
- Volatility: MAPE < 15-20%
- Trading volume: MAPE < 20-30%

## GPU Acceleration

### Enable GPU Support

```bash
# Build with CUDA support
cargo build --release --features cuda

# Build with Metal support (macOS)
cargo build --release --features metal
```

### Use GPU in Training

```bash
neural-trader train-neural \
  --data data.csv \
  --gpu 0 \
  --output model.safetensors
```

### Mixed Precision Training

```bash
neural-trader train-neural \
  --data data.csv \
  --gpu 0 \
  --mixed-precision \
  --output model.safetensors
```

**Benefits**:
- 2-3x faster training
- 50% less memory usage
- Minimal accuracy loss

## Best Practices

### 1. Data Preparation

✅ **DO**:
- Normalize/scale your data
- Remove outliers or handle them appropriately
- Ensure no missing values or NaNs
- Use sufficient historical data (>1000 samples recommended)

❌ **DON'T**:
- Use data with gaps or irregular timestamps
- Mix different scales without normalization
- Include future information in features (data leakage)

### 2. Model Configuration

✅ **DO**:
- Start with default configuration
- Use validation split for model selection
- Enable early stopping to prevent overfitting
- Monitor both training and validation loss

❌ **DON'T**:
- Use very small validation sets (< 10% of data)
- Train without validation
- Ignore diverging validation loss

### 3. Training Process

✅ **DO**:
- Start with shorter training (10-20 epochs) to verify setup
- Use checkpointing for long training runs
- Monitor loss curves for signs of overfitting
- Test on held-out data before production use

❌ **DON'T**:
- Train for too many epochs without early stopping
- Use the same data for training and validation
- Deploy models without proper testing

### 4. Hyperparameter Tuning

✅ **DO**:
- Tune one parameter at a time
- Use cross-validation for robust evaluation
- Keep notes on experiments
- Use grid search or random search systematically

❌ **DON'T**:
- Change multiple parameters simultaneously
- Rely on single train/val split
- Skip validation after tuning

## Troubleshooting

### Loss Not Decreasing

**Possible causes**:
1. Learning rate too low
2. Model too simple for data complexity
3. Data not normalized
4. Gradient vanishing

**Solutions**:
```bash
# Increase learning rate
--lr 0.01

# Increase model capacity
--hidden-size 1024 --n-stacks 4

# Check data preprocessing
# Enable gradient clipping
```

### Loss Diverging (NaN)

**Possible causes**:
1. Learning rate too high
2. Gradient exploding
3. Data issues (NaN, Inf values)

**Solutions**:
```bash
# Decrease learning rate
--lr 0.0001

# Enable/check gradient clipping (enabled by default)
# Inspect data for invalid values
```

### Overfitting

**Symptoms**:
- Training loss decreases but validation loss increases
- Large gap between train and validation loss

**Solutions**:
```bash
# Increase dropout
--dropout 0.3

# Add weight decay
--weight-decay 1e-4

# Reduce model capacity
--hidden-size 256

# Get more training data
```

### Out of Memory (OOM)

**Solutions**:
```bash
# Reduce batch size
--batch-size 16

# Reduce model size
--hidden-size 256 --n-stacks 2

# Use gradient accumulation (not yet implemented)
```

### Slow Training

**Solutions**:
```bash
# Use GPU
--gpu 0

# Enable mixed precision
--mixed-precision

# Increase batch size (if memory allows)
--batch-size 64

# Reduce validation frequency
```

## Example Workflows

### Workflow 1: Quick Experiment

```bash
# Train small model for quick iteration
neural-trader train-neural \
  --data data.csv \
  --target close \
  --epochs 10 \
  --batch-size 32 \
  --hidden-size 128 \
  --output quick_model.safetensors
```

### Workflow 2: Production Model

```bash
# Train production model with best practices
neural-trader train-neural \
  --data large_dataset.parquet \
  --target close \
  --input-size 168 \
  --horizon 24 \
  --epochs 200 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden-size 512 \
  --dropout 0.1 \
  --weight-decay 1e-5 \
  --patience 15 \
  --checkpoint-dir ./production_ckpts \
  --gpu 0 \
  --mixed-precision \
  --output production_model.safetensors
```

### Workflow 3: Hyperparameter Search

```bash
# Grid search over learning rates
for lr in 0.0001 0.001 0.01; do
  neural-trader train-neural \
    --data data.csv \
    --lr $lr \
    --output model_lr_${lr}.safetensors
done

# Grid search over hidden sizes
for hs in 128 256 512 1024; do
  neural-trader train-neural \
    --data data.csv \
    --hidden-size $hs \
    --output model_hs_${hs}.safetensors
done
```

## Next Steps

After training:

1. **Test the model**: Evaluate on held-out test set
2. **Run backtests**: Test in simulated trading environment
3. **Deploy**: Use in paper trading before live trading
4. **Monitor**: Track model performance over time
5. **Retrain**: Update model with new data periodically

## References

- [NHITS Paper](https://arxiv.org/abs/2201.12886)
- [Neural Trader Documentation](../README.md)
- [Model Architecture Details](./NEURAL_MODELS.md)
