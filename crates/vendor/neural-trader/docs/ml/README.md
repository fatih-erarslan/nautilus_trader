# Neural Network / ML Documentation

Complete documentation for neural network training, evaluation, and deployment in neural-trader.

## Quick Start

```bash
# Install dependencies
npm install

# Run complete training pipeline
node examples/ml/complete-training-pipeline.js

# Run validation tests
npm test tests/ml/neural-validation.test.js

# Run performance benchmarks
npm test tests/ml/model-performance.test.js
```

## Documentation Structure

### 📚 Guides

1. **[Neural Network Guide](./neural-network-guide.md)**
   - Comprehensive guide to using neural networks for trading
   - Model architecture explanations
   - Data preparation and feature engineering
   - Training workflows and evaluation
   - Backtesting and production deployment

2. **[Training Best Practices](./training-best-practices.md)**
   - Production-grade training guidelines
   - Model selection strategies
   - Overfitting prevention techniques
   - Performance optimization tips
   - Monitoring and maintenance procedures

3. **[Production Deployment Checklist](./production-deployment-checklist.md)**
   - Pre-deployment requirements
   - Infrastructure setup
   - Monitoring and alerting
   - Risk management procedures
   - Emergency rollback protocols

### 🧪 Tests

1. **[Neural Validation Tests](../../tests/ml/neural-validation.test.js)**
   - Comprehensive function validation
   - Forecast accuracy testing
   - Training convergence validation
   - Evaluation metrics verification
   - Optimization testing
   - Backtest accuracy validation

2. **[Model Performance Tests](../../tests/ml/model-performance.test.js)**
   - GPU vs CPU benchmarks
   - Data size scaling tests
   - Inference latency measurements
   - Memory usage profiling
   - Batch processing efficiency

### 💼 Examples

1. **[Complete Training Pipeline](../../examples/ml/complete-training-pipeline.js)**
   - End-to-end ML workflow
   - Data preparation
   - Multi-model training
   - Evaluation and optimization
   - Backtesting
   - Production readiness check

## Available Neural Functions

### Forecasting

```javascript
const prediction = await neuralForecast(
  'AAPL',     // symbol
  24,         // horizon (hours)
  true,       // use GPU
  0.95        // confidence level
);

// Returns: predictions, confidence intervals, model accuracy
```

### Training

```javascript
const result = await neuralTrain(
  './data/training.csv',  // data path
  'lstm',                 // model type
  100,                    // epochs
  true                    // use GPU
);

// Returns: model ID, training metrics
```

### Evaluation

```javascript
const metrics = await neuralEvaluate(
  modelId,          // trained model ID
  './data/test.csv', // test data
  true              // use GPU
);

// Returns: MAE, RMSE, MAPE, R² score
```

### Optimization

```javascript
const optimized = await neuralOptimize(
  modelId,
  JSON.stringify({
    learning_rate: [0.001, 0.01],
    batch_size: [32, 64]
  }),
  true  // use GPU
);

// Returns: best parameters, best score
```

### Backtesting

```javascript
const backtest = await neuralBacktest(
  modelId,
  '2023-01-01',
  '2023-12-31',
  'SPY',  // benchmark
  true    // use GPU
);

// Returns: total return, Sharpe, drawdown, win rate
```

### Model Status

```javascript
const models = await neuralModelStatus();
// or
const model = await neuralModelStatus(modelId);

// Returns: model type, status, accuracy, creation date
```

## Model Types

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| **GRU** | ⚡⚡⚡ | ⭐⭐⭐ | 💾💾 | Real-time trading |
| **LSTM** | ⚡⚡ | ⭐⭐⭐⭐ | 💾💾💾 | General forecasting |
| **Transformer** | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | Multi-asset portfolios |
| **CNN** | ⚡⚡⚡ | ⭐⭐⭐ | 💾💾 | Pattern recognition |
| **Hybrid** | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | Maximum accuracy |

## Performance Characteristics

### Training Speed (5000 samples, 100 epochs)

- **GRU**: ~30 seconds (GPU) / ~5 minutes (CPU)
- **LSTM**: ~45 seconds (GPU) / ~8 minutes (CPU)
- **Transformer**: ~90 seconds (GPU) / ~15 minutes (CPU)

### Inference Latency

- **Single prediction**: < 100ms (GPU) / < 500ms (CPU)
- **Batch (100 predictions)**: < 500ms (GPU) / < 3 seconds (CPU)

### Memory Requirements

- **GRU**: ~150MB training / ~50MB inference
- **LSTM**: ~250MB training / ~75MB inference
- **Transformer**: ~500MB training / ~150MB inference

## Best Practices Summary

### ✅ Data Quality

1. **Minimum 1000 samples** for reliable training
2. **Regular intervals** with consistent time spacing
3. **Clean data** with outliers handled
4. **Validation split** for overfitting detection

### ✅ Model Selection

1. Start with **GRU** for speed, **LSTM** for accuracy
2. Use **Transformer** for multi-asset strategies
3. Enable **GPU** for datasets > 1000 samples
4. **Ensemble multiple models** for production

### ✅ Training Strategy

1. **100 epochs** as default, adjust based on convergence
2. **Early stopping** to prevent overfitting
3. **Cross-validation** for robust evaluation
4. **Hyperparameter optimization** for best performance

### ✅ Evaluation Criteria

- **R² Score > 0.70** for production deployment
- **Overfitting gap < 10%** (train vs test accuracy)
- **MAE < 5%** of average price
- **Sharpe Ratio > 1.0** in backtesting

### ✅ Production Requirements

1. **Model versioning** and registry
2. **A/B testing** for new models
3. **Performance monitoring** with alerts
4. **Retraining schedule** (weekly/monthly)
5. **Rollback procedure** documented

## Common Issues & Solutions

### Issue: Low Model Accuracy

**Solutions:**
- Increase training data (> 1000 samples)
- Try different model type (Transformer > LSTM > GRU)
- Check data quality (outliers, missing values)
- Optimize hyperparameters
- Add feature engineering

### Issue: Slow Training

**Solutions:**
- Enable GPU (`use_gpu: true`)
- Use faster model (GRU instead of LSTM)
- Reduce batch size
- Decrease model complexity

### Issue: Overfitting

**Solutions:**
- Increase dropout rate (0.2-0.3)
- Add more training data
- Reduce model complexity
- Use regularization (L2 weight decay)
- Enable early stopping

### Issue: High Inference Latency

**Solutions:**
- Enable GPU for inference
- Use smaller model (GRU)
- Batch predictions together
- Cache frequent predictions
- Optimize hardware

## Performance Metrics Reference

### Regression Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference
  - Target: < 5% of mean value
  - Lower is better

- **RMSE (Root Mean Squared Error)**: Penalizes large errors
  - Typically 1.2-1.5x MAE
  - Lower is better

- **MAPE (Mean Absolute Percentage Error)**: Percentage error
  - Target: < 10%
  - Lower is better

- **R² Score**: Explained variance
  - Target: > 0.70 for production
  - Range: -∞ to 1.0, higher is better

### Trading Metrics

- **Sharpe Ratio**: Risk-adjusted returns
  - Target: > 1.0 for production
  - > 2.0 is excellent

- **Max Drawdown**: Largest peak-to-trough decline
  - Target: < 30% acceptable
  - Negative value (e.g., -0.25 = 25% drawdown)

- **Win Rate**: Percentage of profitable trades
  - Target: > 45%
  - > 55% is strong

- **Profit Factor**: Gross profit / gross loss
  - Target: > 1.5
  - > 2.0 is excellent

## Testing Strategy

### Unit Tests
```bash
npm test tests/ml/neural-validation.test.js
```

Tests individual neural functions with various parameters and edge cases.

### Performance Tests
```bash
npm test tests/ml/model-performance.test.js
```

Benchmarks training speed, inference latency, and memory usage.

### Integration Tests
```bash
node examples/ml/complete-training-pipeline.js
```

End-to-end pipeline from data preparation to deployment readiness.

## Deployment Workflow

```
1. Data Preparation
   ├─ Validate quality (> 1000 samples)
   ├─ Handle missing values
   ├─ Remove outliers
   └─ Split train/test

2. Model Training
   ├─ Train multiple models (GRU, LSTM, Transformer)
   ├─ Monitor convergence
   └─ Select best performing

3. Evaluation
   ├─ Test on holdout data
   ├─ Calculate metrics (MAE, RMSE, R²)
   ├─ Check for overfitting
   └─ Cross-validation

4. Optimization
   ├─ Hyperparameter tuning
   ├─ Architecture search
   └─ Ensemble methods

5. Backtesting
   ├─ Historical simulation
   ├─ Calculate Sharpe, drawdown
   └─ Compare to benchmark

6. Production Deployment
   ├─ Version model
   ├─ Setup monitoring
   ├─ Configure alerts
   ├─ A/B testing
   └─ Gradual rollout

7. Monitoring & Maintenance
   ├─ Track accuracy
   ├─ Monitor latency
   ├─ Detect degradation
   └─ Scheduled retraining
```

## Resources

### Documentation
- [Neural Network Guide](./neural-network-guide.md)
- [Training Best Practices](./training-best-practices.md)
- [Production Checklist](./production-deployment-checklist.md)

### Code Examples
- [Complete Pipeline](../../examples/ml/complete-training-pipeline.js)
- [Validation Tests](../../tests/ml/neural-validation.test.js)
- [Performance Tests](../../tests/ml/model-performance.test.js)

### External Resources
- [Deep Learning for Time Series](https://arxiv.org/abs/2004.10240)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/neural-trader/issues)
- **Discord**: [Join our community](https://discord.gg/neural-trader)
- **Documentation**: [Full API reference](../../neural-trader-rust/packages/neural-trader-backend/index.d.ts)

---

**Last Updated**: 2025-11-15
**Version**: 2.1.0
