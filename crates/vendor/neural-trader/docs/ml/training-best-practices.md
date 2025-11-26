# Neural Network Training Best Practices

Production-grade guidelines for training, evaluating, and deploying neural trading models.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Model Selection](#model-selection)
3. [Training Strategy](#training-strategy)
4. [Overfitting Prevention](#overfitting-prevention)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Risk Management](#risk-management)

## Data Preparation

### Data Quality Checklist

#### ✅ Pre-Training Validation

```javascript
async function validateTrainingData(filePath) {
  const data = await loadCSV(filePath);

  const checks = {
    minSamples: data.length >= 1000,
    noMissing: data.every(row => row.value !== null && row.value !== undefined),
    regularIntervals: checkRegularIntervals(data),
    outliers: checkOutliers(data),
    stationarity: checkStationarity(data)
  };

  console.log('Data Quality Report:');
  Object.entries(checks).forEach(([check, passed]) => {
    console.log(`${passed ? '✓' : '✗'} ${check}`);
  });

  return Object.values(checks).every(v => v);
}

function checkRegularIntervals(data) {
  const intervals = [];
  for (let i = 1; i < data.length; i++) {
    const diff = new Date(data[i].timestamp) - new Date(data[i-1].timestamp);
    intervals.push(diff);
  }

  const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const variance = intervals.reduce((sum, val) =>
    sum + Math.pow(val - avgInterval, 2), 0) / intervals.length;

  // Intervals should be consistent (low variance)
  return variance < avgInterval * 0.1;
}

function checkOutliers(data, threshold = 3) {
  const values = data.map(d => d.value);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
  );

  const outliers = values.filter(v => Math.abs(v - mean) > threshold * std);
  return outliers.length < values.length * 0.05; // < 5% outliers
}

function checkStationarity(data) {
  // Simple stationarity check: compare first and second half means
  const mid = Math.floor(data.length / 2);
  const firstHalf = data.slice(0, mid).map(d => d.value);
  const secondHalf = data.slice(mid).map(d => d.value);

  const mean1 = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
  const mean2 = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

  // Means shouldn't differ by more than 20%
  return Math.abs(mean2 - mean1) / mean1 < 0.20;
}
```

### Feature Engineering

#### Technical Indicators

```javascript
function addTechnicalIndicators(data) {
  const enhanced = [...data];

  // Simple Moving Average (SMA)
  for (let i = 20; i < enhanced.length; i++) {
    const window = enhanced.slice(i - 20, i).map(d => d.value);
    enhanced[i].sma_20 = window.reduce((a, b) => a + b) / window.length;
  }

  // Exponential Moving Average (EMA)
  const k = 2 / (20 + 1);
  enhanced[0].ema_20 = enhanced[0].value;
  for (let i = 1; i < enhanced.length; i++) {
    enhanced[i].ema_20 = enhanced[i].value * k + enhanced[i-1].ema_20 * (1 - k);
  }

  // RSI (Relative Strength Index)
  const rsiPeriod = 14;
  for (let i = rsiPeriod; i < enhanced.length; i++) {
    const changes = [];
    for (let j = i - rsiPeriod; j < i; j++) {
      changes.push(enhanced[j+1].value - enhanced[j].value);
    }

    const gains = changes.filter(c => c > 0).reduce((a, b) => a + b, 0) / rsiPeriod;
    const losses = Math.abs(changes.filter(c => c < 0).reduce((a, b) => a + b, 0)) / rsiPeriod;

    const rs = gains / (losses || 1);
    enhanced[i].rsi = 100 - (100 / (1 + rs));
  }

  return enhanced;
}
```

#### Normalization Strategies

```javascript
// Z-score normalization (default)
function zScoreNormalize(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
  );
  return values.map(v => (v - mean) / std);
}

// Min-Max normalization (for bounded outputs)
function minMaxNormalize(values, min = 0, max = 1) {
  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  return values.map(v =>
    min + ((v - dataMin) / (dataMax - dataMin)) * (max - min)
  );
}

// Robust normalization (outlier-resistant)
function robustNormalize(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const median = sorted[Math.floor(sorted.length * 0.5)];
  const iqr = q3 - q1;

  return values.map(v => (v - median) / iqr);
}
```

## Model Selection

### Decision Tree

```
Start
  │
  ├─ Need real-time inference? ─── Yes ──→ GRU
  │                              └─ No
  │                                  │
  ├─ Multiple assets? ─────────── Yes ──→ Transformer
  │                              └─ No
  │                                  │
  ├─ Complex patterns? ───────── Yes ──→ LSTM or Hybrid
  │                              └─ No
  │                                  │
  └─ Simple trends? ─────────── Yes ──→ GRU or CNN
```

### Model Comparison Matrix

| Criteria | GRU | LSTM | Transformer | CNN | Hybrid |
|----------|-----|------|-------------|-----|--------|
| **Training Speed** | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡⚡⚡ | ⚡ |
| **Inference Speed** | ⚡⚡⚡ | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡ |
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory Usage** | 💾💾 | 💾💾💾 | 💾💾💾💾 | 💾💾 | 💾💾💾💾 |
| **Multi-Asset** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Long-Term Deps** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |

### Selection Guide

```javascript
function selectOptimalModel(requirements) {
  const {
    dataSize,
    latencyMs,
    numAssets,
    accuracy,
    budget
  } = requirements;

  // Real-time constraint
  if (latencyMs < 100) {
    return 'gru';
  }

  // Multi-asset portfolio
  if (numAssets > 5) {
    return 'transformer';
  }

  // High accuracy requirement
  if (accuracy === 'highest') {
    return budget === 'unlimited' ? 'hybrid' : 'transformer';
  }

  // Large dataset
  if (dataSize > 100000) {
    return 'lstm';
  }

  // Default: balanced choice
  return 'lstm';
}
```

## Training Strategy

### Progressive Training

```javascript
async function progressiveTraining(dataPath) {
  const stages = [
    { epochs: 20, lr: 0.01, name: 'warm-up' },
    { epochs: 50, lr: 0.001, name: 'main' },
    { epochs: 30, lr: 0.0001, name: 'fine-tune' }
  ];

  let modelId = null;

  for (const stage of stages) {
    console.log(`\n=== Stage: ${stage.name} ===`);

    const result = await backend.neuralTrain(
      dataPath,
      'lstm',
      stage.epochs,
      true
    );

    modelId = result.modelId;

    console.log(`Loss: ${result.finalLoss.toFixed(6)}`);
    console.log(`Accuracy: ${result.validationAccuracy.toFixed(4)}`);

    // Early stop if converged
    if (result.finalLoss < 0.001) {
      console.log('Converged early!');
      break;
    }
  }

  return modelId;
}
```

### Ensemble Training

```javascript
async function trainEnsemble(dataPath, numModels = 5) {
  const modelTypes = ['lstm', 'gru', 'transformer'];
  const models = [];

  for (let i = 0; i < numModels; i++) {
    const modelType = modelTypes[i % modelTypes.length];

    console.log(`Training model ${i+1}/${numModels} (${modelType})...`);

    const result = await backend.neuralTrain(
      dataPath,
      modelType,
      100,
      true
    );

    models.push({
      id: result.modelId,
      type: modelType,
      accuracy: result.validationAccuracy
    });
  }

  // Weight models by accuracy
  const totalAccuracy = models.reduce((sum, m) => sum + m.accuracy, 0);
  models.forEach(m => {
    m.weight = m.accuracy / totalAccuracy;
  });

  return models;
}

async function ensemblePredict(ensemble, symbol, horizon) {
  const predictions = await Promise.all(
    ensemble.map(model => backend.neuralForecast(symbol, horizon, true))
  );

  // Weighted average
  const weightedPredictions = Array(horizon).fill(0);

  for (let i = 0; i < predictions.length; i++) {
    const weight = ensemble[i].weight;
    for (let j = 0; j < horizon; j++) {
      weightedPredictions[j] += predictions[i].predictions[j] * weight;
    }
  }

  return weightedPredictions;
}
```

## Overfitting Prevention

### Early Detection

```javascript
async function detectOverfitting(trainResult, evalResult) {
  const trainAccuracy = trainResult.validationAccuracy;
  const testAccuracy = 1 - (evalResult.mae / 150); // Normalize to 0-1

  const gap = trainAccuracy - testAccuracy;

  console.log('\n=== Overfitting Analysis ===');
  console.log(`Training Accuracy: ${trainAccuracy.toFixed(4)}`);
  console.log(`Test Accuracy: ${testAccuracy.toFixed(4)}`);
  console.log(`Gap: ${gap.toFixed(4)}`);

  if (gap > 0.10) {
    console.log('⚠️  WARNING: Significant overfitting detected');
    console.log('Recommendations:');
    console.log('  1. Increase dropout rate (try 0.3)');
    console.log('  2. Add more training data');
    console.log('  3. Reduce model complexity');
    console.log('  4. Use regularization');
    return 'high';
  } else if (gap > 0.05) {
    console.log('⚠️  Mild overfitting detected');
    return 'mild';
  } else {
    console.log('✓ No significant overfitting');
    return 'none';
  }
}
```

### Regularization Techniques

```javascript
const regularizationConfig = {
  // L2 weight decay
  weightDecay: 0.0001,

  // Dropout rates
  dropout: {
    input: 0.1,
    hidden: 0.2,
    output: 0.1
  },

  // Gradient clipping
  gradientClip: 1.0,

  // Early stopping
  earlyStoppingPatience: 10,

  // Data augmentation
  augmentation: {
    noise: 0.01,
    scaling: 0.05,
    shift: 0.02
  }
};
```

### Cross-Validation Strategy

```javascript
async function kFoldCrossValidation(dataPath, k = 5) {
  const foldSize = Math.floor(totalSamples / k);
  const results = [];

  for (let fold = 0; fold < k; fold++) {
    console.log(`\n=== Fold ${fold + 1}/${k} ===`);

    // Split data (in production, actually split the CSV)
    const trainData = dataPath; // Training fold
    const testData = dataPath;  // Test fold

    // Train
    const trainResult = await backend.neuralTrain(
      trainData,
      'lstm',
      50,
      true
    );

    // Evaluate
    const evalResult = await backend.neuralEvaluate(
      trainResult.modelId,
      testData,
      true
    );

    results.push({
      fold,
      trainAccuracy: trainResult.validationAccuracy,
      testMAE: evalResult.mae,
      testR2: evalResult.r2Score
    });
  }

  // Calculate statistics
  const avgTrainAcc = results.reduce((s, r) => s + r.trainAccuracy, 0) / k;
  const avgTestMAE = results.reduce((s, r) => s + r.testMAE, 0) / k;
  const stdTestMAE = Math.sqrt(
    results.reduce((s, r) => s + Math.pow(r.testMAE - avgTestMAE, 2), 0) / k
  );

  console.log('\n=== Cross-Validation Results ===');
  console.log(`Average Train Accuracy: ${avgTrainAcc.toFixed(4)}`);
  console.log(`Average Test MAE: ${avgTestMAE.toFixed(4)} ± ${stdTestMAE.toFixed(4)}`);
  console.log(`Coefficient of Variation: ${(stdTestMAE / avgTestMAE * 100).toFixed(2)}%`);

  return { avgTrainAcc, avgTestMAE, stdTestMAE, results };
}
```

## Performance Optimization

### GPU Utilization

```javascript
async function optimizeGPUUsage() {
  console.log('=== GPU Optimization Tips ===\n');

  // Batch size optimization
  const batchSizes = [16, 32, 64, 128, 256];
  console.log('Optimal batch sizes by GPU memory:');
  console.log('  4GB VRAM:  batch_size = 16-32');
  console.log('  8GB VRAM:  batch_size = 32-64');
  console.log('  16GB VRAM: batch_size = 64-128');
  console.log('  24GB+ VRAM: batch_size = 128-256');

  // Mixed precision
  console.log('\n✓ Mixed precision training (FP16) enabled by default');
  console.log('  - 2x faster training');
  console.log('  - 50% less memory usage');
  console.log('  - Minimal accuracy impact');

  // Multi-GPU
  console.log('\n⚠️  Multi-GPU support:');
  console.log('  - Currently single-GPU only');
  console.log('  - Multi-GPU coming in v2.2.0');
}
```

### Memory Optimization

```javascript
function calculateMemoryRequirements(config) {
  const {
    modelType,
    hiddenSize,
    numLayers,
    batchSize,
    sequenceLength
  } = config;

  // Rough memory estimates (MB)
  const baseMemory = {
    gru: 50,
    lstm: 75,
    transformer: 150,
    cnn: 40,
    hybrid: 200
  };

  const parameterMemory = hiddenSize * hiddenSize * numLayers * 4 / 1024 / 1024;
  const activationMemory = batchSize * sequenceLength * hiddenSize * 4 / 1024 / 1024;

  const total = baseMemory[modelType] + parameterMemory + activationMemory;

  console.log(`\n=== Memory Requirements ===`);
  console.log(`Model Type: ${modelType}`);
  console.log(`Base: ${baseMemory[modelType]}MB`);
  console.log(`Parameters: ${parameterMemory.toFixed(2)}MB`);
  console.log(`Activations: ${activationMemory.toFixed(2)}MB`);
  console.log(`Total: ~${total.toFixed(2)}MB`);

  return total;
}
```

### Training Speed Tips

```javascript
const speedOptimizations = {
  // Model selection
  fasterModels: ['gru', 'cnn'], // vs lstm, transformer

  // Hyperparameters
  reducedComplexity: {
    hiddenSize: 256,    // vs 512
    numLayers: 2,       // vs 3-4
    batchSize: 64       // vs 32
  },

  // Data
  sampledData: {
    useSubset: true,
    sampleRate: 0.8     // Use 80% of data
  },

  // Hardware
  gpu: {
    enable: true,
    mixedPrecision: true,
    cudnn: true
  },

  // Early stopping
  earlyStop: {
    enabled: true,
    patience: 10,
    minDelta: 0.0001
  }
};

console.log('Speed Optimization Techniques:');
console.log('1. Use GRU instead of LSTM (30% faster)');
console.log('2. Enable GPU and mixed precision (10-100x faster)');
console.log('3. Increase batch size (better GPU utilization)');
console.log('4. Reduce hidden size and layers (faster, less accurate)');
console.log('5. Use early stopping (avoid unnecessary epochs)');
console.log('6. Sample data for hyperparameter search');
```

## Production Deployment

### Model Versioning

```javascript
class ModelRegistry {
  constructor(storageDir = './models') {
    this.storageDir = storageDir;
    this.models = new Map();
  }

  async register(modelId, metadata) {
    const version = Date.now();
    const modelInfo = {
      id: modelId,
      version,
      ...metadata,
      registeredAt: new Date().toISOString()
    };

    this.models.set(modelId, modelInfo);

    // Save to disk
    await fs.writeFile(
      `${this.storageDir}/${modelId}_metadata.json`,
      JSON.stringify(modelInfo, null, 2)
    );

    console.log(`Registered model ${modelId} (v${version})`);
    return modelInfo;
  }

  async getProduction() {
    // Return model with highest accuracy
    const models = Array.from(this.models.values());
    return models.sort((a, b) => b.accuracy - a.accuracy)[0];
  }

  async rollback(modelId) {
    const model = this.models.get(modelId);
    if (!model) throw new Error('Model not found');

    console.log(`Rolling back to model ${modelId}`);
    return model;
  }
}
```

### A/B Testing Framework

```javascript
class ModelABTest {
  constructor(modelA, modelB) {
    this.modelA = modelA;
    this.modelB = modelB;
    this.resultsA = [];
    this.resultsB = [];
    this.trafficSplit = 0.5; // 50/50 split
  }

  async predict(symbol, horizon) {
    const useModelA = Math.random() < this.trafficSplit;
    const modelId = useModelA ? this.modelA : this.modelB;

    const startTime = Date.now();
    const prediction = await backend.neuralForecast(symbol, horizon, true);
    const latency = Date.now() - startTime;

    // Track metrics
    const result = {
      modelId,
      symbol,
      prediction,
      latency,
      timestamp: new Date()
    };

    if (useModelA) {
      this.resultsA.push(result);
    } else {
      this.resultsB.push(result);
    }

    return prediction;
  }

  async analyze() {
    const avgLatencyA = this.resultsA.reduce((s, r) => s + r.latency, 0) / this.resultsA.length;
    const avgLatencyB = this.resultsB.reduce((s, r) => s + r.latency, 0) / this.resultsB.length;

    console.log('\n=== A/B Test Results ===');
    console.log(`Model A: ${this.resultsA.length} predictions, ${avgLatencyA.toFixed(2)}ms avg`);
    console.log(`Model B: ${this.resultsB.length} predictions, ${avgLatencyB.toFixed(2)}ms avg`);

    // Statistical significance test would go here
    return {
      modelA: { count: this.resultsA.length, avgLatency: avgLatencyA },
      modelB: { count: this.resultsB.length, avgLatency: avgLatencyB }
    };
  }
}
```

### Health Monitoring

```javascript
class ModelHealthMonitor {
  constructor(modelId, thresholds) {
    this.modelId = modelId;
    this.thresholds = thresholds || {
      maxLatency: 1000,      // ms
      minAccuracy: 0.70,     // 70%
      maxErrorRate: 0.05     // 5%
    };
    this.metrics = [];
  }

  async check() {
    const metrics = {
      timestamp: new Date(),
      latency: await this.measureLatency(),
      accuracy: await this.measureAccuracy(),
      errorRate: await this.measureErrorRate()
    };

    this.metrics.push(metrics);

    // Check thresholds
    const alerts = [];

    if (metrics.latency > this.thresholds.maxLatency) {
      alerts.push(`High latency: ${metrics.latency}ms > ${this.thresholds.maxLatency}ms`);
    }

    if (metrics.accuracy < this.thresholds.minAccuracy) {
      alerts.push(`Low accuracy: ${metrics.accuracy} < ${this.thresholds.minAccuracy}`);
    }

    if (metrics.errorRate > this.thresholds.maxErrorRate) {
      alerts.push(`High error rate: ${metrics.errorRate} > ${this.thresholds.maxErrorRate}`);
    }

    if (alerts.length > 0) {
      console.error('⚠️  Model Health Alerts:');
      alerts.forEach(alert => console.error(`  - ${alert}`));

      // Trigger alerts (Slack, PagerDuty, etc.)
      await this.sendAlerts(alerts);
    }

    return { metrics, alerts };
  }

  async measureLatency() {
    const start = Date.now();
    await backend.neuralForecast('AAPL', 24, true);
    return Date.now() - start;
  }

  async measureAccuracy() {
    // Compare predictions vs actuals
    // This would require storing predictions and comparing to actual prices
    return 0.85; // Placeholder
  }

  async measureErrorRate() {
    // Track failed predictions
    return 0.02; // Placeholder
  }

  async sendAlerts(alerts) {
    // Integration with alerting systems
    console.log('Sending alerts to monitoring system...');
  }
}
```

## Monitoring & Maintenance

### Retraining Schedule

```javascript
const retrainingStrategy = {
  // When to retrain
  triggers: [
    'accuracy_degradation',  // Accuracy drops > 5%
    'scheduled_weekly',      // Every Monday
    'market_regime_change',  // Volatility spike
    'manual_override'        // Admin trigger
  ],

  // Retraining frequency by market condition
  schedule: {
    normal: '7d',      // Weekly
    volatile: '3d',    // Every 3 days
    crisis: '1d'       // Daily
  },

  // Data window for retraining
  dataWindow: {
    minimum: 1000,     // samples
    optimal: 5000,
    maximum: 50000
  }
};

async function scheduledRetraining(modelId) {
  console.log('Checking if retraining is needed...');

  // Check model performance
  const currentAccuracy = await getCurrentAccuracy(modelId);
  const baselineAccuracy = await getBaselineAccuracy(modelId);

  const degradation = (baselineAccuracy - currentAccuracy) / baselineAccuracy;

  if (degradation > 0.05) {
    console.log(`⚠️  Accuracy degraded by ${(degradation * 100).toFixed(2)}%`);
    console.log('Triggering retraining...');

    // Retrain with latest data
    const newModel = await backend.neuralTrain(
      './data/latest_data.csv',
      'lstm',
      100,
      true
    );

    console.log(`✓ New model trained: ${newModel.modelId}`);
    return newModel.modelId;
  }

  console.log('✓ Model performance is acceptable');
  return modelId;
}
```

## Risk Management

### Position Sizing with Confidence

```javascript
function calculatePositionSize(prediction, bankroll, riskPercent = 0.02) {
  const { predictions, confidenceIntervals, modelAccuracy } = prediction;

  // Expected return
  const currentPrice = predictions[0];
  const futurePrice = predictions[predictions.length - 1];
  const expectedReturn = (futurePrice - currentPrice) / currentPrice;

  // Confidence width (uncertainty)
  const lastInterval = confidenceIntervals[confidenceIntervals.length - 1];
  const uncertainty = (lastInterval.upper - lastInterval.lower) / currentPrice;

  // Adjust position size by confidence
  const basePosition = bankroll * riskPercent;
  const confidenceAdjustment = modelAccuracy * (1 - uncertainty);
  const position = basePosition * confidenceAdjustment;

  console.log('\n=== Position Sizing ===');
  console.log(`Base position (${riskPercent * 100}% risk): $${basePosition.toFixed(2)}`);
  console.log(`Model accuracy: ${(modelAccuracy * 100).toFixed(2)}%`);
  console.log(`Uncertainty: ${(uncertainty * 100).toFixed(2)}%`);
  console.log(`Confidence adjustment: ${(confidenceAdjustment * 100).toFixed(2)}%`);
  console.log(`Final position: $${position.toFixed(2)}`);

  return {
    position,
    expectedReturn,
    uncertainty,
    confidence: modelAccuracy
  };
}
```

### Stop-Loss Automation

```javascript
function calculateDynamicStopLoss(prediction) {
  const { confidenceIntervals, modelAccuracy } = prediction;

  // Use lower confidence bound as stop-loss
  const firstInterval = confidenceIntervals[0];
  const stopLossPercent = (firstInterval.lower - firstInterval.upper) / firstInterval.upper;

  // Adjust by model accuracy
  const adjustedStopLoss = stopLossPercent * (1 + (1 - modelAccuracy));

  console.log(`Stop-loss: ${(adjustedStopLoss * 100).toFixed(2)}%`);

  return adjustedStopLoss;
}
```

---

## Checklist: Pre-Production

- [ ] Data quality validated (> 1000 samples, no missing values)
- [ ] Multiple models trained and compared
- [ ] Cross-validation performed (k ≥ 5)
- [ ] Overfitting checked (train/test gap < 5%)
- [ ] Hyperparameters optimized
- [ ] Backtest results satisfactory (Sharpe > 1.0)
- [ ] Model versioning system in place
- [ ] Monitoring and alerting configured
- [ ] Retraining schedule defined
- [ ] Risk management rules implemented
- [ ] Fallback strategy prepared
- [ ] Documentation complete

---

**Next Steps**: See `complete-training-pipeline.js` for full implementation example.
