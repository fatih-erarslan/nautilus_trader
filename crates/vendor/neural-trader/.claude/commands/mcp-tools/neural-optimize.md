# MCP Tool: neural_optimize

## Overview
Optimize neural model hyperparameters using specified ranges and trials. This tool uses advanced optimization algorithms (Optuna/Ray Tune) to find the best hyperparameter configuration for your neural forecasting models.

## Tool Details
- **Name**: `mcp__ai-news-trader__neural_optimize`
- **Category**: Neural Forecasting Tools
- **GPU Support**: Essential for efficient optimization (100x speedup)
- **Optimization Algorithms**: TPE, CMA-ES, Random Search, Grid Search

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | *required* | ID of the model to optimize |
| `parameter_ranges` | object | *required* | Dictionary of parameters and their search ranges |
| `trials` | integer | `100` | Number of optimization trials to run |
| `optimization_metric` | string | `"mae"` | Metric to optimize: "mae", "rmse", "sharpe", "direction_accuracy" |
| `use_gpu` | boolean | `true` | Enable GPU acceleration |

## Parameter Range Specification

### Supported Parameter Types
```json
{
  "parameter_ranges": {
    // Continuous parameters
    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": true},
    "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5},
    
    // Integer parameters
    "hidden_size": {"type": "int", "low": 128, "high": 1024, "step": 128},
    "num_layers": {"type": "int", "low": 2, "high": 8},
    
    // Categorical parameters
    "optimizer": {"type": "categorical", "choices": ["adam", "adamw", "sgd"]},
    "activation": {"type": "categorical", "choices": ["relu", "gelu", "tanh"]},
    
    // Conditional parameters
    "momentum": {"type": "float", "low": 0.8, "high": 0.99, "condition": "optimizer==sgd"}
  }
}
```

## Return Value Structure
```json
{
  "model_id": "nhits_aapl_v3.2",
  "optimization_results": {
    "best_params": {
      "learning_rate": 0.00032,
      "dropout_rate": 0.15,
      "hidden_size": 512,
      "num_layers": 4,
      "optimizer": "adamw",
      "batch_size": 64,
      "gradient_clip": 1.0
    },
    "best_value": 0.00742,
    "improvement": 0.285,
    "baseline_value": 0.01032
  },
  "study_statistics": {
    "n_trials": 100,
    "n_complete": 98,
    "n_pruned": 2,
    "best_trial_number": 73,
    "optimization_time_hours": 4.2,
    "avg_trial_time_minutes": 2.5
  },
  "performance_history": [
    {"trial": 1, "value": 0.01032, "params": {...}},
    {"trial": 2, "value": 0.00986, "params": {...}},
    {"trial": 73, "value": 0.00742, "params": {...}}
  ],
  "validation_metrics": {
    "mae": 0.00742,
    "rmse": 0.01125,
    "mape": 0.0068,
    "direction_accuracy": 0.738,
    "sharpe_ratio": 2.34
  },
  "importance_scores": {
    "learning_rate": 0.42,
    "hidden_size": 0.28,
    "dropout_rate": 0.15,
    "num_layers": 0.08,
    "optimizer": 0.07
  },
  "optimization_config": {
    "algorithm": "TPE",
    "pruner": "MedianPruner",
    "sampler_seed": 42,
    "gpu_devices": ["cuda:0", "cuda:1"],
    "parallel_trials": 4
  }
}
```

## Examples

### Example 1: Basic Learning Rate Optimization
```bash
# Optimize learning rate and batch size
claude --mcp ai-news-trader "Optimize learning rate for my AAPL model"

# The tool will be called as:
mcp__ai-news-trader__neural_optimize({
  "model_id": "nhits_aapl_v3",
  "parameter_ranges": {
    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": true},
    "batch_size": {"type": "int", "low": 16, "high": 128, "step": 16}
  },
  "trials": 50,
  "optimization_metric": "mae",
  "use_gpu": true
})
```

### Example 2: Comprehensive Architecture Search
```bash
# Full architecture optimization
claude --mcp ai-news-trader "Optimize all hyperparameters for my portfolio model"

# The tool will be called as:
mcp__ai-news-trader__neural_optimize({
  "model_id": "nbeats_portfolio_v2",
  "parameter_ranges": {
    "learning_rate": {"type": "float", "low": 0.00001, "high": 0.01, "log": true},
    "hidden_size": {"type": "int", "low": 256, "high": 2048, "step": 256},
    "num_blocks": {"type": "int", "low": 3, "high": 10},
    "num_layers": {"type": "int", "low": 2, "high": 6},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.1}
  },
  "trials": 200,
  "optimization_metric": "sharpe",
  "use_gpu": true
})
```

### Example 3: Trading-Focused Optimization
```bash
# Optimize for trading performance
claude --mcp ai-news-trader "Optimize my model for maximum trading profits"

# The tool will be called as:
mcp__ai-news-trader__neural_optimize({
  "model_id": "deepar_trading_v1",
  "parameter_ranges": {
    "forecast_threshold": {"type": "float", "low": 0.005, "high": 0.03},
    "confidence_threshold": {"type": "float", "low": 0.7, "high": 0.95},
    "position_size_factor": {"type": "float", "low": 0.5, "high": 2.0},
    "stop_loss": {"type": "float", "low": 0.01, "high": 0.05},
    "take_profit": {"type": "float", "low": 0.02, "high": 0.10}
  },
  "trials": 150,
  "optimization_metric": "sharpe",
  "use_gpu": true
})
```

### Example 4: Quick Parameter Tuning
```bash
# Fast optimization for specific parameters
claude --mcp ai-news-trader "Quick tune dropout and regularization"

# The tool will be called as:
mcp__ai-news-trader__neural_optimize({
  "model_id": "nhits_spy_v4",
  "parameter_ranges": {
    "dropout_rate": {"type": "float", "low": 0.1, "high": 0.4},
    "l2_regularization": {"type": "float", "low": 0.0001, "high": 0.01, "log": true}
  },
  "trials": 30,
  "optimization_metric": "mae",
  "use_gpu": true
})
```

### Example 5: Multi-Objective Optimization
```bash
# Optimize for both accuracy and speed
claude --mcp ai-news-trader "Optimize model for accuracy while keeping inference under 10ms"

# The tool will be called as:
mcp__ai-news-trader__neural_optimize({
  "model_id": "nhits_realtime_v2",
  "parameter_ranges": {
    "hidden_size": {"type": "int", "low": 128, "high": 512, "step": 64},
    "num_layers": {"type": "int", "low": 2, "high": 4},
    "use_attention": {"type": "categorical", "choices": [true, false]},
    "quantization": {"type": "categorical", "choices": ["none", "int8", "fp16"]}
  },
  "trials": 100,
  "optimization_metric": "mae",
  "use_gpu": true
})
```

## GPU Acceleration Notes

### Optimization Performance
| Trials | CPU Time | GPU Time | GPU Speedup |
|--------|----------|----------|-------------|
| 10 | 2 hours | 5 min | 24x |
| 50 | 10 hours | 20 min | 30x |
| 100 | 20 hours | 35 min | 34x |
| 500 | 4 days | 2.5 hours | 38x |

### Parallel Trial Execution
- **Single GPU**: 1-2 parallel trials
- **Multi-GPU**: 4-8 parallel trials
- **Distributed**: 16+ parallel trials
- **Auto-scaling**: Based on available resources

## Optimization Algorithms

### TPE (Tree-structured Parzen Estimator)
- **Best for**: General optimization, mixed parameter types
- **Convergence**: Fast, typically 50-100 trials
- **Strengths**: Handles categorical parameters well

### CMA-ES (Covariance Matrix Adaptation)
- **Best for**: Continuous parameters only
- **Convergence**: Very fast for small search spaces
- **Strengths**: Excellent for fine-tuning

### Random Search
- **Best for**: Initial exploration, baseline
- **Convergence**: Slow but unbiased
- **Strengths**: Simple, parallelizable

### Bayesian Optimization
- **Best for**: Expensive evaluations
- **Convergence**: Efficient with few trials
- **Strengths**: Uncertainty quantification

## Advanced Optimization Strategies

### Pruning Strategies
```python
# Early stopping for bad trials
pruner_configs = {
    "MedianPruner": {"n_startup_trials": 5, "n_warmup_steps": 10},
    "PercentilePruner": {"percentile": 25.0, "n_startup_trials": 5},
    "HyperbandPruner": {"min_resource": 1, "max_resource": 100}
}
```

### Multi-Stage Optimization
```python
# Stage 1: Coarse search
coarse_ranges = {
    "learning_rate": {"low": 0.0001, "high": 0.1, "log": true},
    "hidden_size": {"low": 128, "high": 2048, "step": 256}
}

# Stage 2: Fine search around best values
fine_ranges = {
    "learning_rate": {"low": best_lr * 0.5, "high": best_lr * 2.0},
    "hidden_size": {"low": best_hs - 128, "high": best_hs + 128}
}
```

### Conditional Parameters
```python
# Architecture-dependent parameters
if architecture == "transformer":
    ranges["num_heads"] = {"type": "int", "low": 4, "high": 16}
    ranges["attention_dropout"] = {"type": "float", "low": 0.0, "high": 0.3}
```

## Optimization Metrics

### Forecasting Metrics
- **MAE**: Best for general accuracy
- **RMSE**: Penalizes large errors
- **MAPE**: Scale-independent comparison
- **Direction Accuracy**: For trading signals

### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Calmar Ratio**: Return vs drawdown
- **Profit Factor**: Gross profit/loss ratio
- **Win Rate**: Percentage of profitable trades

### Custom Metrics
```python
def custom_metric(y_true, y_pred, trades):
    accuracy = mean_absolute_error(y_true, y_pred)
    profits = calculate_trading_pnl(trades)
    return 0.7 * accuracy + 0.3 * profits
```

## Best Practices

### Parameter Range Selection
1. **Start Wide**: Begin with broad ranges
2. **Log Scale**: Use for learning rates, regularization
3. **Reasonable Bounds**: Avoid extreme values
4. **Domain Knowledge**: Incorporate prior experience
5. **Iterative Refinement**: Narrow ranges based on results

### Trial Budget Allocation
| Model Complexity | Recommended Trials | Time Estimate (GPU) |
|-----------------|-------------------|-------------------|
| Simple (< 1M params) | 50-100 | 30-60 min |
| Medium (1-10M params) | 100-200 | 1-3 hours |
| Large (> 10M params) | 200-500 | 3-8 hours |

### Validation Strategy
1. **Hold-out Set**: Separate from train/val
2. **Time-based Split**: Respect temporal order
3. **Cross-validation**: For robust estimates
4. **Walk-forward**: For time series
5. **Multiple Seeds**: Average over runs

## Common Optimization Patterns

### Learning Rate Schedule
```python
# Optimize both initial LR and schedule
parameter_ranges = {
    "initial_lr": {"type": "float", "low": 0.0001, "high": 0.01},
    "lr_schedule": {"type": "categorical", 
                   "choices": ["constant", "cosine", "exponential"]},
    "lr_decay": {"type": "float", "low": 0.9, "high": 0.99}
}
```

### Regularization Suite
```python
# Comprehensive regularization tuning
parameter_ranges = {
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
    "gradient_clip": {"type": "float", "low": 0.5, "high": 5.0},
    "label_smoothing": {"type": "float", "low": 0.0, "high": 0.1}
}
```

## Troubleshooting

### Issue: "Optimization not improving"
- Check parameter ranges aren't too narrow
- Increase number of trials
- Try different optimization algorithm
- Verify metric calculation is correct

### Issue: "Trials failing/pruned"
- Reduce learning rate upper bound
- Check for numerical instability
- Increase pruner patience
- Verify data quality

### Issue: "Optimization too slow"
- Enable GPU acceleration
- Reduce model size for search
- Use aggressive pruning
- Parallelize trials

## Integration Examples

### Automated Retraining Pipeline
```python
# Weekly optimization routine
if model_performance_degraded():
    best_params = neural_optimize(
        model_id=current_model,
        parameter_ranges=standard_ranges,
        trials=100
    )
    new_model = neural_train(
        data=recent_data,
        **best_params
    )
    if validate_model(new_model):
        deploy_model(new_model)
```

### Ensemble Optimization
```python
# Optimize ensemble weights
parameter_ranges = {
    f"weight_{model}": {"type": "float", "low": 0.0, "high": 1.0}
    for model in ensemble_models
}
# Constraint: weights sum to 1
```

## Related Tools
- `neural_train`: Train models with optimized parameters
- `neural_evaluate`: Validate optimization results
- `neural_backtest`: Test optimized trading strategies
- `optimize_strategy`: Traditional strategy optimization
- `neural_model_status`: Monitor optimization progress