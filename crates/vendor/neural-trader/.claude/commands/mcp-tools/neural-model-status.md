# MCP Tool: neural_model_status

## Overview
Get status and information about neural models - specific model or all models. This tool provides comprehensive insights into model health, performance metrics, training history, and deployment readiness.

## Tool Details
- **Name**: `mcp__ai-news-trader__neural_model_status`
- **Category**: Neural Forecasting Tools
- **GPU Support**: Not required (metadata query only)
- **Real-time Monitoring**: Live status updates for deployed models

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | `null` | Specific model ID to query, or null for all models |

## Return Value Structure

### Single Model Response
```json
{
  "model_id": "nhits_aapl_v3.2",
  "status": "deployed",
  "model_info": {
    "architecture": "NHITS",
    "version": "3.2",
    "created_date": "2024-12-15T14:30:00Z",
    "last_updated": "2024-12-27T09:00:00Z",
    "parameters": 2548320,
    "file_size_mb": 48.5,
    "compression": "quantized_int8"
  },
  "training_info": {
    "dataset": "aapl_2020_2024_enhanced.csv",
    "train_samples": 1250000,
    "val_samples": 250000,
    "features": ["open", "high", "low", "close", "volume", "rsi", "macd", "sentiment"],
    "epochs_trained": 120,
    "best_epoch": 87,
    "training_time_hours": 2.4,
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "hardware": "4x NVIDIA A100"
  },
  "performance_metrics": {
    "validation": {
      "mae": 0.821,
      "rmse": 1.145,
      "mape": 0.0079,
      "direction_accuracy": 0.725
    },
    "production": {
      "last_24h_mae": 0.892,
      "last_7d_mae": 0.856,
      "last_30d_mae": 0.834,
      "predictions_served": 458290,
      "avg_latency_ms": 8.2
    }
  },
  "deployment_info": {
    "environment": "production",
    "endpoint": "https://api.neural-trader.com/v1/models/nhits_aapl_v3.2",
    "replicas": 3,
    "load_balancer": "round_robin",
    "auto_scaling": true,
    "health_check": "passing",
    "uptime_percentage": 99.98
  },
  "usage_statistics": {
    "daily_predictions": 15420,
    "unique_users": 234,
    "api_calls_24h": 45678,
    "cache_hit_rate": 0.82,
    "error_rate": 0.0002
  }
}
```

### All Models Response
```json
{
  "total_models": 12,
  "models": [
    {
      "model_id": "nhits_aapl_v3.2",
      "status": "deployed",
      "architecture": "NHITS",
      "created": "2024-12-15",
      "performance": {
        "mape": 0.0079,
        "sharpe_ratio": 2.15
      },
      "usage_24h": 15420
    },
    {
      "model_id": "nbeats_portfolio_v2.1",
      "status": "testing",
      "architecture": "NBEATSx",
      "created": "2024-12-20",
      "performance": {
        "mape": 0.0091,
        "sharpe_ratio": 1.92
      },
      "usage_24h": 0
    }
  ],
  "summary": {
    "deployed": 8,
    "testing": 3,
    "archived": 1,
    "total_predictions_24h": 125840,
    "average_latency_ms": 9.8,
    "total_storage_gb": 2.4
  }
}
```

## Model Status Types
- **training**: Currently being trained
- **validating**: Undergoing validation testing
- **testing**: In staging/testing environment
- **deployed**: Active in production
- **paused**: Temporarily disabled
- **failed**: Training or deployment failed
- **archived**: Retired but preserved
- **updating**: Being retrained/updated

## Examples

### Example 1: Check Specific Model Status
```bash
# Get detailed status of a specific model
claude --mcp ai-news-trader "Check status of my AAPL neural model"

# The tool will be called as:
mcp__ai-news-trader__neural_model_status({
  "model_id": "nhits_aapl_v3.2"
})
```

### Example 2: List All Models
```bash
# Get overview of all neural models
claude --mcp ai-news-trader "Show me all my neural trading models"

# The tool will be called as:
mcp__ai-news-trader__neural_model_status({
  "model_id": null
})
```

### Example 3: Production Health Check
```bash
# Check health of deployed models
claude --mcp ai-news-trader "Are all my production models healthy?"

# The tool will be called as:
mcp__ai-news-trader__neural_model_status({
  "model_id": null
})
# Claude will filter for deployed models and check health
```

### Example 4: Performance Monitoring
```bash
# Monitor model performance degradation
claude --mcp ai-news-trader "Check if any models are underperforming"

# The tool will be called as:
mcp__ai-news-trader__neural_model_status({
  "model_id": null
})
# Claude will compare validation vs production metrics
```

### Example 5: Model Selection
```bash
# Find best model for specific use case
claude --mcp ai-news-trader "Which model should I use for NASDAQ stocks?"

# The tool will be called as:
mcp__ai-news-trader__neural_model_status({
  "model_id": null
})
# Claude will analyze models and recommend based on performance
```

## Model Health Indicators

### Green (Healthy)
- Production MAE within 10% of validation
- Latency < 50ms
- Error rate < 0.1%
- Uptime > 99.9%
- Recent predictions consistent

### Yellow (Warning)
- Production MAE 10-25% worse than validation
- Latency 50-100ms
- Error rate 0.1-1%
- Uptime 95-99.9%
- Some prediction anomalies

### Red (Critical)
- Production MAE >25% worse than validation
- Latency > 100ms
- Error rate > 1%
- Uptime < 95%
- Systematic prediction failures

## Performance Monitoring

### Key Metrics to Track
1. **Accuracy Drift**: Compare production vs validation metrics
2. **Latency Trends**: Monitor inference speed over time
3. **Error Rates**: Track prediction failures
4. **Usage Patterns**: Identify peak load times
5. **Resource Utilization**: CPU/GPU/Memory usage

### Automated Alerts
```python
# Alert thresholds
if production_mae > validation_mae * 1.2:
    alert("Model degradation detected")
if latency_p99 > 100:
    alert("High latency warning")
if error_rate > 0.01:
    alert("Elevated error rate")
```

## Model Lifecycle Management

### Stages
1. **Development**: Initial training and validation
2. **Testing**: A/B testing in staging environment
3. **Deployment**: Gradual rollout to production
4. **Monitoring**: Continuous performance tracking
5. **Maintenance**: Regular retraining and updates
6. **Retirement**: Archive when superseded

### Version Control
```
Model Naming Convention:
{architecture}_{asset}_{version}_{variant}

Examples:
- nhits_aapl_v3.2_standard
- nbeats_portfolio_v2.1_enhanced
- deepar_spy_v1.0_experimental
```

## Storage and Resource Management

### Model Storage
| Model Type | Size (Uncompressed) | Size (Quantized) | RAM Required |
|------------|--------------------|--------------------|--------------|
| NHITS | 50-100 MB | 12-25 MB | 200 MB |
| NBEATSx | 80-150 MB | 20-38 MB | 300 MB |
| DeepAR | 120-200 MB | 30-50 MB | 400 MB |
| TFT | 200-400 MB | 50-100 MB | 600 MB |

### Deployment Resources
- **CPU Instance**: 1-2 models per core
- **GPU Instance**: 10-20 models per GPU
- **Memory**: 2GB base + model requirements
- **Disk**: 10GB for model storage and caching

## Best Practices

### Model Management
1. **Naming Convention**: Use consistent model IDs
2. **Documentation**: Maintain training logs and configs
3. **Versioning**: Semantic versioning (major.minor.patch)
4. **Backups**: Keep last 3 versions of each model
5. **Testing**: Always validate before deployment

### Monitoring Strategy
1. **Real-time Dashboards**: Track key metrics
2. **Daily Reports**: Performance summaries
3. **Weekly Reviews**: Trend analysis
4. **Monthly Audits**: Full model evaluation
5. **Quarterly Retraining**: Update with new data

### Deployment Strategy
```bash
# Canary Deployment
1. Deploy to 5% of traffic
2. Monitor for 24 hours
3. If metrics stable, increase to 25%
4. Monitor for 48 hours
5. If successful, full deployment
```

## Common Issues and Solutions

### Issue: "Model not found"
```bash
# List all models to check ID
neural_model_status()
# Verify model path
ls /models/
```

### Issue: "Performance degradation"
```bash
# Check data drift
analyze_input_distribution()
# Retrain with recent data
neural_train --incremental
```

### Issue: "High latency"
```bash
# Check model size
neural_model_status(model_id)
# Consider quantization
optimize_model --quantize
```

## Integration Examples

### Automated Model Selection
```python
# Select best model for current market
models = neural_model_status()
best_model = max(models, 
    key=lambda m: m["performance"]["sharpe_ratio"])
use_model(best_model["model_id"])
```

### Health Check Integration
```python
# Continuous monitoring
while True:
    status = neural_model_status(production_model)
    if status["health"] != "green":
        switch_to_backup_model()
        alert_team(status)
    sleep(300)  # Check every 5 minutes
```

### Model Comparison Dashboard
```python
# Compare multiple models
models = neural_model_status()
for model in models:
    print(f"{model['id']}: {model['performance']}")
plot_performance_comparison(models)
```

## Related Tools
- `neural_train`: Create new models
- `neural_evaluate`: Test model performance
- `neural_optimize`: Improve model parameters
- `neural_forecast`: Use models for predictions
- `neural_backtest`: Validate trading performance