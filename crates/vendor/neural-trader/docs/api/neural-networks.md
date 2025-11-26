# Neural Networks API

**Category:** Machine Learning  
**Functions:** 7  
**Version:** 3.0.0

High-performance GPU-accelerated neural network training, prediction, and evaluation functions.

---

## Table of Contents

1. [neuralTrain](#neuraltrain)
2. [neuralPredict](#neuralpredict)
3. [neuralBacktest](#neuralbacktest)
4. [neuralEvaluate](#neuralevaluate)
5. [neuralForecast](#neuralforecast)
6. [neuralOptimize](#neuraloptimize)
7. [neuralModelStatus](#neuralmodelstatus)

---

## neuralTrain()

Train a neural network model for price prediction or strategy optimization.

### Signature

```typescript
neuralTrain(config: NeuralTrainConfig): Promise<TrainingResult>
```

### Parameters

- **config** (Object) - Training configuration
  - **modelType** (string) - Model architecture: `'lstm'`, `'gru'`, `'transformer'`, `'feedforward'`
  - **data** (Array<Array<number>>) - Training data matrix [samples × features]
  - **labels** (Array<number>) - Target values for supervised learning
  - **epochs** (number) - Number of training epochs (default: 100)
  - **batchSize** (number) - Mini-batch size (default: 32)
  - **learningRate** (number) - Learning rate (default: 0.001)
  - **layers** (Array<number>) - Hidden layer sizes (e.g., `[64, 32, 16]`)
  - **activation** (string) - Activation function: `'relu'`, `'tanh'`, `'sigmoid'` (default: 'relu')
  - **optimizer** (string) - Optimizer: `'adam'`, `'sgd'`, `'rmsprop'` (default: 'adam')
  - **validation_split** (number) - Validation data fraction (0.0-1.0, default: 0.2)
  - **early_stopping** (boolean) - Enable early stopping (default: true)
  - **patience** (number) - Early stopping patience (default: 10)
  - **gpu** (boolean) - Use GPU acceleration if available (default: true)

### Returns

**Promise<TrainingResult>** - Training results and model ID

```typescript
interface TrainingResult {
  modelId: string;          // Unique model identifier
  finalLoss: number;        // Final training loss
  validationLoss: number;   // Final validation loss
  epochs: number;           // Epochs completed
  trainingTime: number;     // Training duration (ms)
  accuracy: number;         // Model accuracy (0-1)
  metrics: {
    mse: number;            // Mean squared error
    mae: number;            // Mean absolute error
    r2: number;             // R-squared score
  };
  modelPath: string;        // Path to saved model
}
```

### Example: Basic LSTM Training

```javascript
const nt = require('neural-trader');

// Prepare time series data
const data = [
  [100, 101, 102, 103, 104],  // Sample 1: prices for 5 days
  [105, 104, 103, 102, 101],  // Sample 2
  [101, 102, 103, 104, 105],  // Sample 3
  // ... more samples
];

const labels = [105, 100, 106];  // Next day's price for each sample

// Train LSTM model
const result = await nt.neuralTrain({
  modelType: 'lstm',
  data,
  labels,
  epochs: 100,
  batchSize: 32,
  learningRate: 0.001,
  layers: [64, 32],
  activation: 'relu',
  optimizer: 'adam',
  validation_split: 0.2,
  early_stopping: true,
  patience: 10,
  gpu: true
});

console.log('Model trained:', result.modelId);
console.log('Final loss:', result.finalLoss);
console.log('Validation loss:', result.validationLoss);
console.log('Accuracy:', result.accuracy);
console.log('Training time:', result.trainingTime, 'ms');
```

### Example: Advanced Transformer Model

```javascript
// Train transformer for multi-step ahead forecasting
const result = await nt.neuralTrain({
  modelType: 'transformer',
  data: priceSequences,      // [N × 50 × 5] - 50 timesteps, 5 features
  labels: futureLabels,      // [N × 10] - predict next 10 timesteps
  epochs: 200,
  batchSize: 64,
  learningRate: 0.0005,
  layers: [128, 64, 32],
  activation: 'relu',
  optimizer: 'adam',
  validation_split: 0.15,
  early_stopping: true,
  patience: 15,
  gpu: true
});
```

### Error Handling

```javascript
try {
  const result = await nt.neuralTrain(config);
} catch (error) {
  if (error.message.includes('GPU not available')) {
    // Fallback to CPU training
    config.gpu = false;
    const result = await nt.neuralTrain(config);
  } else if (error.message.includes('Invalid data shape')) {
    console.error('Data shape mismatch. Expected: [samples, features]');
  } else if (error.message.includes('Insufficient training data')) {
    console.error('Need at least 100 samples for reliable training');
  }
}
```

### Performance Notes

- **GPU Acceleration:** 10-50x faster than CPU training
- **Memory Usage:** ~500MB for LSTM with 100K samples
- **Training Time:** 
  - LSTM: ~30s for 100 epochs on GPU
  - Transformer: ~2min for 200 epochs on GPU
  - Feedforward: ~10s for 100 epochs on GPU

### See Also

- [neuralPredict()](#neuralpredict) - Make predictions with trained model
- [neuralEvaluate()](#neuralevaluate) - Evaluate model performance
- [neuralOptimize()](#neuraloptimize) - Hyperparameter optimization

---

## neuralPredict()

Make predictions using a trained neural network model.

### Signature

```typescript
neuralPredict(modelId: string, input: number[][]): Promise<PredictionResult>
```

### Parameters

- **modelId** (string) - Model identifier from `neuralTrain()`
- **input** (Array<Array<number>>) - Input data matrix [samples × features]

### Returns

**Promise<PredictionResult>** - Predictions and confidence scores

```typescript
interface PredictionResult {
  predictions: number[];      // Predicted values
  confidence: number[];       // Confidence scores (0-1)
  uncertainty: number[];      // Prediction uncertainty
  inferenceTime: number;      // Inference duration (ms)
}
```

### Example: Single Prediction

```javascript
// Predict next day's price
const input = [[100, 101, 102, 103, 104]];  // Last 5 days

const result = await nt.neuralPredict(modelId, input);

console.log('Predicted price:', result.predictions[0]);
console.log('Confidence:', result.confidence[0]);
console.log('Uncertainty:', result.uncertainty[0]);
```

### Example: Batch Predictions

```javascript
// Predict for multiple sequences
const inputs = [
  [100, 101, 102, 103, 104],
  [105, 104, 103, 102, 101],
  [101, 102, 103, 104, 105]
];

const result = await nt.neuralPredict(modelId, inputs);

result.predictions.forEach((pred, i) => {
  console.log(`Sample ${i}: ${pred} (confidence: ${result.confidence[i]})`);
});
```

### Performance Notes

- **GPU Inference:** 1000+ predictions/second
- **CPU Inference:** 100-500 predictions/second
- **Latency:** <10ms per prediction on GPU

### See Also

- [neuralTrain()](#neuraltrain)
- [neuralForecast()](#neuralforecast)

---

## neuralBacktest()

Backtest a neural network-based trading strategy.

### Signature

```typescript
neuralBacktest(config: NeuralBacktestConfig): Promise<BacktestResult>
```

### Parameters

- **config** (Object) - Backtest configuration
  - **modelId** (string) - Trained model ID
  - **symbol** (string) - Trading symbol
  - **startDate** (string) - Start date (ISO format)
  - **endDate** (string) - End date (ISO format)
  - **initialCapital** (number) - Starting capital
  - **predictionThreshold** (number) - Min confidence for trade (0-1, default: 0.6)
  - **positionSize** (number) - Position size per trade (default: 1.0)
  - **stopLoss** (number) - Stop loss percentage (default: 0.05)
  - **takeProfit** (number) - Take profit percentage (default: 0.10)

### Returns

**Promise<BacktestResult>** - Backtest performance metrics

```typescript
interface BacktestResult {
  totalReturn: number;        // Total return (%)
  sharpeRatio: number;        // Sharpe ratio
  maxDrawdown: number;        // Max drawdown (%)
  winRate: number;            // Win rate (0-1)
  totalTrades: number;        // Number of trades
  profitableTrades: number;   // Winning trades
  averageReturn: number;      // Average return per trade
  finalCapital: number;       // Ending capital
  trades: Trade[];            // Individual trade records
}
```

### Example

```javascript
const result = await nt.neuralBacktest({
  modelId: trainedModelId,
  symbol: 'AAPL',
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  initialCapital: 10000,
  predictionThreshold: 0.7,
  positionSize: 0.1,
  stopLoss: 0.05,
  takeProfit: 0.10
});

console.log('Total Return:', result.totalReturn, '%');
console.log('Sharpe Ratio:', result.sharpeRatio);
console.log('Max Drawdown:', result.maxDrawdown, '%');
console.log('Win Rate:', result.winRate);
console.log('Total Trades:', result.totalTrades);
```

### See Also

- [backtestStrategy()](./strategy-backtest.md#backteststrategy)

---

## neuralEvaluate()

Evaluate model performance on test data.

### Signature

```typescript
neuralEvaluate(modelId: string, testData: number[][], testLabels: number[]): Promise<EvaluationResult>
```

### Parameters

- **modelId** (string) - Model to evaluate
- **testData** (Array<Array<number>>) - Test data matrix
- **testLabels** (Array<number>) - True labels

### Returns

**Promise<EvaluationResult>** - Performance metrics

```typescript
interface EvaluationResult {
  mse: number;              // Mean squared error
  mae: number;              // Mean absolute error
  rmse: number;             // Root mean squared error
  r2: number;               // R-squared score
  accuracy: number;         // Classification accuracy (if applicable)
  confusionMatrix: number[][];  // Confusion matrix
}
```

### Example

```javascript
const evaluation = await nt.neuralEvaluate(
  modelId,
  testData,
  testLabels
);

console.log('MSE:', evaluation.mse);
console.log('MAE:', evaluation.mae);
console.log('R²:', evaluation.r2);
console.log('Accuracy:', evaluation.accuracy);
```

---

## neuralForecast()

Generate multi-step ahead forecasts.

### Signature

```typescript
neuralForecast(modelId: string, input: number[][], steps: number): Promise<ForecastResult>
```

### Parameters

- **modelId** (string) - Trained model ID
- **input** (Array<Array<number>>) - Historical data
- **steps** (number) - Number of steps to forecast

### Returns

**Promise<ForecastResult>** - Forecast values and intervals

```typescript
interface ForecastResult {
  forecast: number[];           // Forecasted values
  lowerBound: number[];         // Lower confidence interval
  upperBound: number[];         // Upper confidence interval
  confidence: number;           // Overall confidence (0-1)
}
```

### Example

```javascript
// Forecast next 10 days
const forecast = await nt.neuralForecast(
  modelId,
  historicalPrices,
  10
);

console.log('Forecast:', forecast.forecast);
console.log('95% CI:', forecast.lowerBound, '-', forecast.upperBound);
```

---

## neuralOptimize()

Perform hyperparameter optimization.

### Signature

```typescript
neuralOptimize(config: OptimizationConfig): Promise<OptimizationResult>
```

### Parameters

- **config** (Object) - Optimization configuration
  - **modelType** (string) - Model architecture
  - **data** (Array<Array<number>>) - Training data
  - **labels** (Array<number>) - Target labels
  - **searchSpace** (Object) - Hyperparameter ranges
  - **trials** (number) - Number of trials (default: 100)
  - **metric** (string) - Optimization metric: `'mse'`, `'mae'`, `'accuracy'`

### Returns

**Promise<OptimizationResult>** - Best hyperparameters and performance

```typescript
interface OptimizationResult {
  bestParams: object;         // Best hyperparameters found
  bestScore: number;          // Best metric score
  trials: TrialResult[];      // All trial results
  modelId: string;            // Best model ID
}
```

### Example

```javascript
const result = await nt.neuralOptimize({
  modelType: 'lstm',
  data: trainingData,
  labels: trainingLabels,
  searchSpace: {
    layers: [[32, 16], [64, 32], [128, 64]],
    learningRate: [0.0001, 0.001, 0.01],
    batchSize: [16, 32, 64],
    activation: ['relu', 'tanh']
  },
  trials: 50,
  metric: 'mse'
});

console.log('Best params:', result.bestParams);
console.log('Best MSE:', result.bestScore);
```

---

## neuralModelStatus()

Get status and metadata for a trained model.

### Signature

```typescript
neuralModelStatus(modelId: string): Promise<ModelStatus>
```

### Parameters

- **modelId** (string) - Model identifier

### Returns

**Promise<ModelStatus>** - Model information

```typescript
interface ModelStatus {
  modelId: string;
  modelType: string;
  createdAt: string;
  parameters: number;       // Total trainable parameters
  size: number;             // Model size (bytes)
  architecture: object;     // Layer configuration
  performance: object;      // Training metrics
}
```

### Example

```javascript
const status = await nt.neuralModelStatus(modelId);

console.log('Model type:', status.modelType);
console.log('Parameters:', status.parameters);
console.log('Size:', status.size / 1024 / 1024, 'MB');
console.log('Created:', status.createdAt);
```

---

## Best Practices

### 1. Data Preparation

```javascript
// Normalize data before training
function normalizeData(data) {
  const mean = data.reduce((a, b) => a + b) / data.length;
  const std = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length);
  return data.map(x => (x - mean) / std);
}

const normalizedData = data.map(sample => normalizeData(sample));
```

### 2. Train-Val-Test Split

```javascript
// Split data: 70% train, 15% val, 15% test
const trainEnd = Math.floor(data.length * 0.7);
const valEnd = Math.floor(data.length * 0.85);

const trainData = data.slice(0, trainEnd);
const valData = data.slice(trainEnd, valEnd);
const testData = data.slice(valEnd);
```

### 3. Model Selection

- **LSTM:** Best for time series with long-term dependencies
- **GRU:** Faster than LSTM, similar performance
- **Transformer:** Best for long sequences (>100 timesteps)
- **Feedforward:** Simple patterns, fastest training

### 4. Hyperparameter Tuning

```javascript
// Start with these defaults
const defaultConfig = {
  layers: [64, 32],
  learningRate: 0.001,
  batchSize: 32,
  activation: 'relu',
  optimizer: 'adam'
};

// Then optimize with neuralOptimize()
```

---

## Common Issues

### Out of Memory

```javascript
// Reduce batch size or model size
config.batchSize = 16;
config.layers = [32, 16];  // Smaller layers
```

### Overfitting

```javascript
// Add regularization
config.dropout = 0.3;
config.early_stopping = true;
config.validation_split = 0.2;
```

### Poor Performance

```javascript
// Try different architectures
config.modelType = 'transformer';  // Instead of LSTM
config.layers = [128, 64, 32];     // Deeper network
config.epochs = 200;                // More training
```

---

## GPU Requirements

| Model Type | Min GPU Memory | Recommended |
|-----------|----------------|-------------|
| Feedforward | 2GB | 4GB |
| LSTM | 4GB | 8GB |
| GRU | 4GB | 8GB |
| Transformer | 8GB | 16GB |

---

## Performance Benchmarks

| Operation | CPU | GPU (RTX 4090) | Speedup |
|-----------|-----|----------------|---------|
| Train LSTM (100 epochs) | 5min | 10s | 30x |
| Predict (1000 samples) | 2s | 0.05s | 40x |
| Optimize (50 trials) | 4h | 8min | 30x |

---

## Related Documentation

- [Strategy & Backtest API](./strategy-backtest.md)
- [Risk Management](./risk-management.md)
- [Neural Networks Guide](../guides/neural-networks-guide.md)

---

**Last Updated:** 2025-11-17  
**Version:** 3.0.0
