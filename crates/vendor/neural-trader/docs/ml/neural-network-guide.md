# Neural Network Trading Guide

Complete guide to using neural networks for financial forecasting and trading with neural-trader.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Preparation](#data-preparation)
4. [Training Workflow](#training-workflow)
5. [Model Evaluation](#model-evaluation)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Backtesting](#backtesting)
8. [Production Deployment](#production-deployment)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

Neural-trader provides GPU-accelerated neural network models for time series forecasting in financial markets. The system supports multiple architectures optimized for different market conditions.

### Available Models

| Model | Description | Best For | Training Speed | Accuracy |
|-------|-------------|----------|----------------|----------|
| **LSTM** | Long Short-Term Memory | General purpose, trending markets | Medium | High |
| **GRU** | Gated Recurrent Unit | Faster alternative to LSTM | Fast | Medium-High |
| **Transformer** | Attention-based architecture | Complex patterns, multi-asset | Slow | Very High |
| **CNN** | Convolutional Neural Network | Pattern recognition | Fast | Medium |
| **Hybrid** | Combined architecture | Maximum accuracy | Slow | Highest |

### Key Features

- ✅ **GPU Acceleration**: 10-100x faster training with CUDA support
- ✅ **Multiple Architectures**: LSTM, GRU, Transformer, CNN, Hybrid
- ✅ **Confidence Intervals**: Probabilistic forecasts with uncertainty quantification
- ✅ **Hyperparameter Optimization**: Automated tuning with Bayesian optimization
- ✅ **Real-time Inference**: Sub-millisecond prediction latency
- ✅ **Backtesting Integration**: Historical strategy validation

## Model Architecture

### LSTM (Long Short-Term Memory)

```javascript
const backend = require('@rUv/neural-trader-backend');

// Train LSTM model
const result = await backend.neuralTrain(
  './data/historical_prices.csv',
  'lstm',
  100, // epochs
  true  // use GPU
);

console.log(`Model ID: ${result.modelId}`);
console.log(`Validation Accuracy: ${result.validationAccuracy}`);
```

**Architecture Details:**
- Input: Historical price sequences (default: 168 timesteps)
- Hidden layers: 3 layers × 512 units
- Dropout: 0.1 for regularization
- Output: Multi-horizon forecasts with confidence intervals

**When to Use:**
- Trending markets with clear momentum
- Medium to long-term predictions (days to weeks)
- When you need good accuracy without extreme complexity

### GRU (Gated Recurrent Unit)

```javascript
// GRU is faster and uses less memory than LSTM
const result = await backend.neuralTrain(
  './data/historical_prices.csv',
  'gru',
  100,
  true
);
```

**Advantages:**
- 30% faster training than LSTM
- Lower memory footprint
- Better for real-time applications

**When to Use:**
- High-frequency trading with latency constraints
- Resource-constrained environments
- Similar performance to LSTM with less complexity

### Transformer

```javascript
// Transformer for complex multi-asset patterns
const result = await backend.neuralTrain(
  './data/multi_asset_prices.csv',
  'transformer',
  150,
  true // GPU highly recommended
);
```

**Advantages:**
- Attention mechanism captures long-range dependencies
- Best for multi-asset correlation learning
- State-of-the-art accuracy

**When to Use:**
- Portfolio optimization
- Multi-asset strategies
- When you have significant compute resources

## Data Preparation

### Data Format Requirements

Neural-trader expects CSV files with the following structure:

```csv
timestamp,value
2023-01-01T00:00:00Z,150.25
2023-01-01T01:00:00Z,150.32
2023-01-01T02:00:00Z,149.98
...
```

### Data Quality Checklist

✅ **Timestamp Format**: ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)
✅ **Regular Intervals**: Consistent time spacing (hourly, daily, etc.)
✅ **No Missing Values**: Fill gaps with forward-fill or interpolation
✅ **Outlier Handling**: Remove or cap extreme values
✅ **Minimum Samples**: At least 1000 data points recommended
✅ **Train/Validation Split**: Automatic 80/20 split

### Data Preprocessing Script

```javascript
const fs = require('fs').promises;

async function preprocessData(rawDataPath, outputPath) {
  // Load raw data
  const rawData = await fs.readFile(rawDataPath, 'utf-8');
  const lines = rawData.split('\n');

  // Parse and clean
  const cleaned = ['timestamp,value'];
  let prevValue = null;

  for (let i = 1; i < lines.length; i++) {
    const [timestamp, value] = lines[i].split(',');

    if (!timestamp || !value) continue;

    // Forward-fill missing values
    const cleanValue = parseFloat(value) || prevValue || 0;

    // Remove outliers (>3 std deviations)
    if (Math.abs(cleanValue - prevValue) < 3 * calculateStd(values)) {
      cleaned.push(`${timestamp},${cleanValue.toFixed(2)}`);
      prevValue = cleanValue;
    }
  }

  await fs.writeFile(outputPath, cleaned.join('\n'));
  console.log(`Preprocessed ${cleaned.length - 1} samples`);
}

// Calculate rolling standard deviation
function calculateStd(values, window = 100) {
  const recent = values.slice(-window);
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recent.length;
  return Math.sqrt(variance);
}
```

## Training Workflow

### Basic Training

```javascript
const backend = require('@rUv/neural-trader-backend');

async function trainModel() {
  console.log('Starting training...');

  const result = await backend.neuralTrain(
    './data/AAPL_hourly.csv',
    'lstm',
    100, // epochs
    true  // use GPU
  );

  console.log('\n=== Training Complete ===');
  console.log(`Model ID: ${result.modelId}`);
  console.log(`Model Type: ${result.modelType}`);
  console.log(`Training Time: ${result.trainingTimeMs}ms`);
  console.log(`Final Loss: ${result.finalLoss.toFixed(6)}`);
  console.log(`Validation Accuracy: ${result.validationAccuracy.toFixed(4)}`);

  return result.modelId;
}
```

### Advanced Training with Monitoring

```javascript
async function trainWithMonitoring() {
  const modelTypes = ['lstm', 'gru', 'transformer'];
  const results = {};

  for (const modelType of modelTypes) {
    console.log(`\nTraining ${modelType.toUpperCase()}...`);

    const startTime = Date.now();
    const startMemory = process.memoryUsage().heapUsed / 1024 / 1024;

    try {
      const result = await backend.neuralTrain(
        './data/training_data.csv',
        modelType,
        100,
        true
      );

      const duration = Date.now() - startTime;
      const memoryUsed = (process.memoryUsage().heapUsed / 1024 / 1024) - startMemory;

      results[modelType] = {
        ...result,
        duration,
        memoryUsed
      };

      console.log(`✓ ${modelType}: ${result.validationAccuracy.toFixed(4)} accuracy`);
      console.log(`  Time: ${duration}ms, Memory: ${memoryUsed.toFixed(2)}MB`);

    } catch (error) {
      console.error(`✗ ${modelType} failed:`, error.message);
    }
  }

  // Find best model
  const bestModel = Object.entries(results)
    .sort((a, b) => b[1].validationAccuracy - a[1].validationAccuracy)[0];

  console.log(`\n🏆 Best Model: ${bestModel[0].toUpperCase()}`);
  console.log(`   Accuracy: ${bestModel[1].validationAccuracy.toFixed(4)}`);

  return bestModel[1].modelId;
}
```

### Early Stopping and Checkpoints

The training system automatically implements:

- **Early Stopping**: Training stops if validation loss doesn't improve for 10 epochs
- **Checkpoints**: Model state saved every 10 epochs
- **Learning Rate Scheduling**: Automatic reduction on plateaus
- **Gradient Clipping**: Prevents exploding gradients

## Model Evaluation

### Comprehensive Evaluation

```javascript
async function evaluateModel(modelId, testDataPath) {
  // Run evaluation
  const metrics = await backend.neuralEvaluate(
    modelId,
    testDataPath,
    true // use GPU
  );

  console.log('\n=== Model Evaluation ===');
  console.log(`Test Samples: ${metrics.testSamples}`);
  console.log(`MAE (Mean Absolute Error): ${metrics.mae.toFixed(4)}`);
  console.log(`RMSE (Root Mean Squared Error): ${metrics.rmse.toFixed(4)}`);
  console.log(`MAPE (Mean Absolute Percentage Error): ${(metrics.mape * 100).toFixed(2)}%`);
  console.log(`R² Score: ${metrics.r2Score.toFixed(4)}`);

  // Interpret results
  if (metrics.r2Score > 0.8) {
    console.log('✓ Excellent model fit');
  } else if (metrics.r2Score > 0.6) {
    console.log('⚠ Good model fit');
  } else {
    console.log('✗ Poor model fit - consider retraining');
  }

  // Check for overfitting
  if (metrics.mape < 0.05) {
    console.log('⚠ Warning: Possible overfitting (MAPE too low)');
  }

  return metrics;
}
```

### Metric Interpretation

| Metric | Formula | Interpretation | Good Value |
|--------|---------|----------------|------------|
| **MAE** | Σ\|y - ŷ\| / n | Average absolute error | < 5% of mean |
| **RMSE** | √(Σ(y - ŷ)² / n) | Penalizes large errors | < 8% of mean |
| **MAPE** | Σ\|y - ŷ\| / y / n | Percentage error | < 10% |
| **R²** | 1 - SS_res / SS_tot | Explained variance | > 0.7 |

### Cross-Validation

```javascript
async function crossValidate(dataPath, modelType, folds = 5) {
  const results = [];

  for (let fold = 0; fold < folds; fold++) {
    console.log(`\nFold ${fold + 1}/${folds}`);

    // Note: In production, you would split data into folds
    // For now, we train on full data
    const trainResult = await backend.neuralTrain(
      dataPath,
      modelType,
      50, // Fewer epochs for CV
      true
    );

    const evalResult = await backend.neuralEvaluate(
      trainResult.modelId,
      dataPath, // Would use test fold
      true
    );

    results.push({
      fold,
      accuracy: trainResult.validationAccuracy,
      mae: evalResult.mae,
      r2: evalResult.r2Score
    });
  }

  // Calculate statistics
  const avgAccuracy = results.reduce((sum, r) => sum + r.accuracy, 0) / folds;
  const avgMAE = results.reduce((sum, r) => sum + r.mae, 0) / folds;
  const stdMAE = Math.sqrt(
    results.reduce((sum, r) => sum + Math.pow(r.mae - avgMAE, 2), 0) / folds
  );

  console.log('\n=== Cross-Validation Results ===');
  console.log(`Average Accuracy: ${avgAccuracy.toFixed(4)} ± ${stdMAE.toFixed(4)}`);
  console.log(`Average MAE: ${avgMAE.toFixed(4)}`);

  return { avgAccuracy, avgMAE, stdMAE, results };
}
```

## Hyperparameter Optimization

### Basic Optimization

```javascript
async function optimizeHyperparameters(modelId) {
  const paramRanges = JSON.stringify({
    learning_rate: [0.0001, 0.001, 0.01],
    batch_size: [16, 32, 64, 128],
    hidden_size: [128, 256, 512, 1024],
    num_layers: [1, 2, 3, 4],
    dropout: [0.0, 0.1, 0.2, 0.3]
  });

  console.log('Starting hyperparameter optimization...');

  const result = await backend.neuralOptimize(
    modelId,
    paramRanges,
    true // use GPU
  );

  const bestParams = JSON.parse(result.bestParams);

  console.log('\n=== Optimization Results ===');
  console.log(`Trials Completed: ${result.trialsCompleted}`);
  console.log(`Optimization Time: ${result.optimizationTimeMs}ms`);
  console.log(`Best Score: ${result.bestScore.toFixed(4)}`);
  console.log('\nBest Parameters:');
  console.log(JSON.stringify(bestParams, null, 2));

  return bestParams;
}
```

### Grid Search vs Bayesian Optimization

```javascript
async function compareOptimizationStrategies(modelId) {
  // Grid search: exhaustive but slow
  const gridParams = JSON.stringify({
    learning_rate: [0.001, 0.01],
    batch_size: [32, 64],
    hidden_size: [256, 512]
  });

  console.log('Running grid search...');
  const gridStart = Date.now();
  const gridResult = await backend.neuralOptimize(modelId, gridParams, true);
  const gridTime = Date.now() - gridStart;

  // Bayesian optimization: smarter sampling
  const bayesianParams = JSON.stringify({
    learning_rate: [0.0001, 0.01],
    batch_size: [16, 128],
    hidden_size: [128, 1024],
    num_layers: [1, 4]
  });

  console.log('Running Bayesian optimization...');
  const bayesStart = Date.now();
  const bayesResult = await backend.neuralOptimize(modelId, bayesianParams, true);
  const bayesTime = Date.now() - bayesStart;

  console.log('\n=== Comparison ===');
  console.log(`Grid Search: ${gridResult.bestScore.toFixed(4)} (${gridTime}ms)`);
  console.log(`Bayesian: ${bayesResult.bestScore.toFixed(4)} (${bayesTime}ms)`);
  console.log(`Speedup: ${(gridTime / bayesTime).toFixed(2)}x`);
}
```

## Backtesting

### Basic Backtest

```javascript
async function backtestStrategy(modelId) {
  const result = await backend.neuralBacktest(
    modelId,
    '2023-01-01',
    '2023-12-31',
    'SPY', // benchmark
    true   // use GPU
  );

  console.log('\n=== Backtest Results ===');
  console.log(`Period: ${result.startDate} to ${result.endDate}`);
  console.log(`Total Return: ${(result.totalReturn * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(2)}`);
  console.log(`Max Drawdown: ${(result.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`Win Rate: ${(result.winRate * 100).toFixed(2)}%`);
  console.log(`Total Trades: ${result.totalTrades}`);

  // Evaluate strategy quality
  evaluateBacktestResults(result);

  return result;
}

function evaluateBacktestResults(result) {
  console.log('\n=== Strategy Evaluation ===');

  // Sharpe Ratio
  if (result.sharpeRatio > 2.0) {
    console.log('✓ Excellent risk-adjusted returns (Sharpe > 2.0)');
  } else if (result.sharpeRatio > 1.0) {
    console.log('⚠ Good risk-adjusted returns (Sharpe > 1.0)');
  } else {
    console.log('✗ Poor risk-adjusted returns (Sharpe < 1.0)');
  }

  // Max Drawdown
  if (result.maxDrawdown > -0.20) {
    console.log('✓ Acceptable drawdown (< 20%)');
  } else if (result.maxDrawdown > -0.40) {
    console.log('⚠ High drawdown (20-40%)');
  } else {
    console.log('✗ Excessive drawdown (> 40%)');
  }

  // Win Rate
  if (result.winRate > 0.55) {
    console.log('✓ Strong win rate (> 55%)');
  } else if (result.winRate > 0.45) {
    console.log('⚠ Moderate win rate (45-55%)');
  } else {
    console.log('✗ Low win rate (< 45%)');
  }
}
```

### Walk-Forward Analysis

```javascript
async function walkForwardAnalysis(dataPath, modelType) {
  const periods = [
    { train: '2022-01-01', trainEnd: '2022-06-30', test: '2022-07-01', testEnd: '2022-12-31' },
    { train: '2022-07-01', trainEnd: '2022-12-31', test: '2023-01-01', testEnd: '2023-06-30' },
    { train: '2023-01-01', trainEnd: '2023-06-30', test: '2023-07-01', testEnd: '2023-12-31' },
  ];

  const results = [];

  for (const period of periods) {
    console.log(`\nTraining: ${period.train} to ${period.trainEnd}`);
    console.log(`Testing: ${period.test} to ${period.testEnd}`);

    // Train on training period
    const trainResult = await backend.neuralTrain(dataPath, modelType, 50, true);

    // Test on out-of-sample period
    const backtestResult = await backend.neuralBacktest(
      trainResult.modelId,
      period.test,
      period.testEnd,
      'SPY',
      true
    );

    results.push(backtestResult);

    console.log(`Return: ${(backtestResult.totalReturn * 100).toFixed(2)}%`);
    console.log(`Sharpe: ${backtestResult.sharpeRatio.toFixed(2)}`);
  }

  // Aggregate results
  const avgReturn = results.reduce((sum, r) => sum + r.totalReturn, 0) / results.length;
  const avgSharpe = results.reduce((sum, r) => sum + r.sharpeRatio, 0) / results.length;

  console.log('\n=== Walk-Forward Results ===');
  console.log(`Average Return: ${(avgReturn * 100).toFixed(2)}%`);
  console.log(`Average Sharpe: ${avgSharpe.toFixed(2)}`);

  return { avgReturn, avgSharpe, results };
}
```

## Production Deployment

See [Production Deployment Checklist](#production-deployment-checklist) for complete deployment guide.

### Real-Time Prediction Pipeline

```javascript
class NeuralTradingEngine {
  constructor(modelId) {
    this.modelId = modelId;
    this.predictionCache = new Map();
    this.cacheTimeout = 60000; // 1 minute
  }

  async predict(symbol, horizon) {
    const cacheKey = `${symbol}_${horizon}`;
    const cached = this.predictionCache.get(cacheKey);

    // Use cache if fresh
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.prediction;
    }

    // Generate new prediction
    const prediction = await backend.neuralForecast(symbol, horizon, true);

    // Cache result
    this.predictionCache.set(cacheKey, {
      prediction,
      timestamp: Date.now()
    });

    return prediction;
  }

  async generateSignal(symbol) {
    const prediction = await this.predict(symbol, 24);

    const current = prediction.predictions[0];
    const future = prediction.predictions[prediction.predictions.length - 1];
    const change = (future - current) / current;

    // Generate trading signal
    if (change > 0.02) {
      return { action: 'BUY', confidence: Math.abs(change), prediction };
    } else if (change < -0.02) {
      return { action: 'SELL', confidence: Math.abs(change), prediction };
    } else {
      return { action: 'HOLD', confidence: 0, prediction };
    }
  }
}
```

## Best Practices

### ✅ Data Best Practices

1. **Minimum Sample Size**: Use at least 1000 data points
2. **Regular Intervals**: Ensure consistent time spacing
3. **Quality Over Quantity**: Clean data beats more data
4. **Feature Engineering**: Add technical indicators if helpful
5. **Normalization**: Data is automatically normalized

### ✅ Training Best Practices

1. **Start Simple**: Begin with GRU or LSTM before trying Transformer
2. **GPU Usage**: Always enable GPU for models > 1000 samples
3. **Epoch Count**: Start with 50-100 epochs, increase if needed
4. **Monitor Validation**: Watch for overfitting
5. **Save Model ID**: Store model IDs for production use

### ✅ Evaluation Best Practices

1. **Multiple Metrics**: Don't rely on single metric
2. **Out-of-Sample Testing**: Always test on unseen data
3. **Walk-Forward Analysis**: Validate temporal robustness
4. **Compare to Baseline**: Beat simple moving average
5. **Monitor Degradation**: Check model performance decay

### ✅ Production Best Practices

1. **Model Versioning**: Track model IDs and performance
2. **Retraining Schedule**: Retrain weekly/monthly
3. **Performance Monitoring**: Track live prediction accuracy
4. **Fallback Strategy**: Have backup model or rule-based system
5. **Risk Management**: Use confidence intervals for position sizing

## Troubleshooting

### Common Issues

**Issue**: Model accuracy is low
**Solutions**:
- Increase training data (need > 1000 samples)
- Try different model type (Transformer > LSTM > GRU)
- Check for data quality issues
- Increase training epochs
- Optimize hyperparameters

**Issue**: Training is slow
**Solutions**:
- Enable GPU (`use_gpu: true`)
- Use GRU instead of LSTM
- Reduce batch size
- Decrease hidden layer size

**Issue**: Predictions are too uncertain
**Solutions**:
- Increase training data
- Reduce forecast horizon
- Use ensemble of models
- Check market volatility

**Issue**: Overfitting on training data
**Solutions**:
- Increase dropout rate
- Reduce model complexity
- Add more training data
- Use early stopping

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Symbol cannot be empty" | Empty symbol parameter | Provide valid symbol string |
| "Horizon exceeds maximum" | Horizon > 365 | Reduce horizon to ≤ 365 |
| "Model not found" | Invalid model ID | Check model ID exists |
| "Training data not found" | File path incorrect | Verify file path |
| "Confidence level must be between 0 and 1" | Invalid confidence | Use 0.80-0.99 |

### Performance Tuning

```javascript
// GPU Performance
// - Enable CUDA for 10-100x speedup
// - Use batch size 32-64 for GPU
// - Increase hidden size on GPU

// Memory Optimization
// - Use GRU instead of LSTM (30% less memory)
// - Reduce batch size if OOM
// - Process predictions in batches

// Latency Optimization
// - Cache predictions (1-minute TTL)
// - Use smaller models for real-time
// - Batch multiple symbols together
```

## Next Steps

1. **Try the Examples**: See `/examples/ml/complete-training-pipeline.js`
2. **Read Best Practices**: See `/docs/ml/training-best-practices.md`
3. **Production Deployment**: See deployment checklist below
4. **Advanced Features**: Explore ensemble methods and custom architectures

---

**Need Help?** Check our [GitHub Issues](https://github.com/yourusername/neural-trader/issues) or [Discord Community](https://discord.gg/neural-trader).
