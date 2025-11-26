# @neural-trader/neural

[![npm version](https://img.shields.io/npm/v/@neural-trader/neural.svg)](https://www.npmjs.com/package/@neural-trader/neural)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)
[![Build Status](https://img.shields.io/github/workflow/status/ruvnet/neural-trader/CI)](https://github.com/ruvnet/neural-trader)
[![Downloads](https://img.shields.io/npm/dm/@neural-trader/neural.svg)](https://www.npmjs.com/package/@neural-trader/neural)

Advanced neural network models for time-series forecasting in financial markets. Powered by Rust with native Node.js bindings, this package provides state-of-the-art architectures including LSTM with Attention, Transformers, and N-HiTS for accurate price prediction with confidence intervals.

## Features

- **State-of-the-Art Architectures**: LSTM with Attention, Transformer, N-HiTS (Neural Hierarchical Interpolation for Time Series)
- **GPU Acceleration**: Optional GPU support for 10-100x faster training
- **Confidence Intervals**: Probabilistic forecasting with upper/lower bounds
- **Multi-Horizon Prediction**: Forecast multiple time steps ahead
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Model Persistence**: Save and load trained models for production use
- **Batch Prediction**: Process multiple inputs efficiently
- **Rust Performance**: Native implementation delivers blazing-fast training and inference

## Installation

```bash
npm install @neural-trader/neural @neural-trader/core
```

**GPU Support**: For GPU acceleration, ensure CUDA 11+ is installed on your system.

## Quick Start

```typescript
import { NeuralModel } from '@neural-trader/neural';
import type { ModelConfig, TrainingConfig } from '@neural-trader/core';

// Configure neural model
const modelConfig: ModelConfig = {
  modelType: 'LSTMAttention',
  inputSize: 10,      // Number of features
  horizon: 5,         // Predict 5 steps ahead
  hiddenSize: 128,    // Hidden layer size
  numLayers: 3,       // Number of layers
  dropout: 0.2,       // Dropout rate
  learningRate: 0.001 // Learning rate
};

// Create model
const model = new NeuralModel(modelConfig);

// Prepare training data (normalized price returns + features)
const data = [/* input sequences */];
const targets = [/* target values */];

// Training configuration
const trainingConfig: TrainingConfig = {
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  earlyStoppingPatience: 10,
  useGpu: true
};

// Train model
const metrics = await model.train(data, targets, trainingConfig);
console.log('Training complete!');
metrics.forEach(m => {
  console.log(`Epoch ${m.epoch}: Loss=${m.trainLoss.toFixed(4)}, Val Loss=${m.valLoss.toFixed(4)}`);
});

// Make prediction
const inputData = [/* recent sequence */];
const prediction = await model.predict(inputData);
console.log('Predictions:', prediction.predictions);
console.log('95% CI:', prediction.lowerBound, '-', prediction.upperBound);

// Save model for later use
await model.save('models/my-model.pt');
```

## In-Depth Usage

### Model Architectures

#### LSTM with Attention

Best for capturing long-term dependencies with attention mechanism:

```typescript
import { NeuralModel } from '@neural-trader/neural';

const lstmConfig: ModelConfig = {
  modelType: 'LSTMAttention',
  inputSize: 20,      // 20 features (price, volume, indicators)
  horizon: 5,         // Predict 5 days ahead
  hiddenSize: 256,    // Large hidden size for complex patterns
  numLayers: 4,       // Deep network
  dropout: 0.3,       // Regularization
  learningRate: 0.0005
};

const model = new NeuralModel(lstmConfig);

// LSTM excels at:
// - Sequential data with temporal dependencies
// - Long-term pattern recognition
// - Market regime detection
// - Volatility forecasting
```

#### Transformer

State-of-the-art architecture for parallel processing:

```typescript
const transformerConfig: ModelConfig = {
  modelType: 'Transformer',
  inputSize: 15,
  horizon: 10,        // Longer horizons work well
  hiddenSize: 512,    // Large hidden size
  numLayers: 6,       // Deep transformer
  dropout: 0.1,
  learningRate: 0.0001 // Lower learning rate
};

const model = new NeuralModel(transformerConfig);

// Transformers excel at:
// - Long-range dependencies
// - Multi-variate time series
// - Cross-asset correlations
// - High-frequency data
```

#### N-HiTS

Neural Hierarchical Interpolation for Time Series:

```typescript
const nhitsConfig: ModelConfig = {
  modelType: 'NHITS',
  inputSize: 10,
  horizon: 20,        // Very long horizons
  hiddenSize: 128,
  numLayers: 3,
  dropout: 0.2,
  learningRate: 0.001
};

const model = new NeuralModel(nhitsConfig);

// N-HiTS excels at:
// - Very long horizon forecasting
// - Trend and seasonality capture
// - Hierarchical pattern learning
// - Efficient training
```

### Training Neural Models

```typescript
import { NeuralModel } from '@neural-trader/neural';

async function trainPricePredictor() {
  // Prepare features: normalized returns + technical indicators
  const features = [];
  const targets = [];

  for (let i = 60; i < marketBars.length - 5; i++) {
    // 60-day lookback window
    const window = marketBars.slice(i - 60, i);

    // Feature engineering
    const input = [
      ...window.map(b => (b.close - b.open) / b.open), // Returns
      ...calculateSMA(window, 20),                      // SMA
      ...calculateRSI(window, 14),                      // RSI
      ...calculateVolume(window)                        // Volume features
    ];

    // Target: 5-day ahead returns
    const futureReturns = marketBars.slice(i, i + 5)
      .map(b => (b.close - marketBars[i - 1].close) / marketBars[i - 1].close);

    features.push(input);
    targets.push(futureReturns);
  }

  // Training configuration with early stopping
  const trainingConfig: TrainingConfig = {
    epochs: 200,
    batchSize: 64,
    validationSplit: 0.2,
    earlyStoppingPatience: 15,  // Stop if no improvement for 15 epochs
    useGpu: true
  };

  const model = new NeuralModel({
    modelType: 'LSTMAttention',
    inputSize: features[0].length,
    horizon: 5,
    hiddenSize: 256,
    numLayers: 4,
    dropout: 0.3,
    learningRate: 0.0005
  });

  console.log('Starting training...');
  const metrics = await model.train(features, targets, trainingConfig);

  // Analyze training progress
  console.log('\nTraining Progress:');
  metrics.forEach((m, i) => {
    if (i % 10 === 0) {  // Print every 10 epochs
      console.log(`Epoch ${m.epoch}:`);
      console.log(`  Train Loss: ${m.trainLoss.toFixed(6)} | MAE: ${m.trainMae.toFixed(6)}`);
      console.log(`  Val Loss: ${m.valLoss.toFixed(6)} | MAE: ${m.valMae.toFixed(6)}`);
    }
  });

  // Save trained model
  await model.save('models/price-predictor-lstm.pt');
  console.log('\nModel saved successfully!');

  return model;
}
```

### Making Predictions

```typescript
import type { PredictionResult } from '@neural-trader/core';

async function makePredictions(model: NeuralModel) {
  // Prepare recent data
  const recentBars = marketBars.slice(-60);  // Last 60 bars
  const inputData = [
    ...recentBars.map(b => (b.close - b.open) / b.open),
    ...calculateSMA(recentBars, 20),
    ...calculateRSI(recentBars, 14),
    ...calculateVolume(recentBars)
  ];

  // Get prediction with confidence intervals
  const prediction: PredictionResult = await model.predict(inputData);

  console.log('5-Day Price Forecast:');
  prediction.predictions.forEach((pred, i) => {
    const lower = prediction.lowerBound[i];
    const upper = prediction.upperBound[i];
    console.log(`Day ${i + 1}: ${(pred * 100).toFixed(2)}% (95% CI: ${(lower * 100).toFixed(2)}% - ${(upper * 100).toFixed(2)}%)`);
  });

  // Calculate expected price levels
  const currentPrice = recentBars[recentBars.length - 1].close;
  console.log('\nExpected Price Levels:');
  prediction.predictions.forEach((pred, i) => {
    const expectedPrice = currentPrice * (1 + pred);
    const lowerPrice = currentPrice * (1 + prediction.lowerBound[i]);
    const upperPrice = currentPrice * (1 + prediction.upperBound[i]);
    console.log(`Day ${i + 1}: $${expectedPrice.toFixed(2)} ($${lowerPrice.toFixed(2)} - $${upperPrice.toFixed(2)})`);
  });

  return prediction;
}
```

### Batch Predictions

```typescript
import { BatchPredictor } from '@neural-trader/neural';

async function batchPredictions() {
  const batchPredictor = new BatchPredictor();

  // Load multiple models for ensemble
  const model1 = new NeuralModel(config1);
  await model1.load('models/lstm-model.pt');

  const model2 = new NeuralModel(config2);
  await model2.load('models/transformer-model.pt');

  const model3 = new NeuralModel(config3);
  await model3.load('models/nhits-model.pt');

  // Add models to batch predictor
  await batchPredictor.addModel(model1);
  await batchPredictor.addModel(model2);
  await batchPredictor.addModel(model3);

  // Prepare inputs for multiple symbols
  const inputs = [
    prepareFeatures('AAPL'),
    prepareFeatures('MSFT'),
    prepareFeatures('GOOGL')
  ];

  // Get predictions for all inputs
  const predictions = await batchPredictor.predictBatch(inputs);

  // Ensemble averaging
  predictions.forEach((pred, i) => {
    console.log(`Symbol ${i + 1}:`);
    console.log('Predictions:', pred.predictions);
  });
}
```

### Model Persistence

```typescript
async function modelLifecycle() {
  // Train and save
  const model = new NeuralModel(config);
  await model.train(data, targets, trainingConfig);
  await model.save('models/production-model.pt');
  console.log('Model saved');

  // Later: Load and use
  const productionModel = new NeuralModel(config);
  await productionModel.load('models/production-model.pt');

  // Get model information
  const info = await productionModel.getInfo();
  console.log('Model Info:', info);

  // Make predictions
  const prediction = await productionModel.predict(newData);
}
```

### Real-Time Trading Integration

```typescript
import { NeuralModel } from '@neural-trader/neural';
import { StrategyRunner } from '@neural-trader/strategies';

class NeuralTradingStrategy {
  private model: NeuralModel;
  private featureBuffer: number[][] = [];

  async initialize() {
    // Load trained model
    this.model = new NeuralModel({
      modelType: 'LSTMAttention',
      inputSize: 100,
      horizon: 1,
      hiddenSize: 256,
      numLayers: 4,
      dropout: 0.3,
      learningRate: 0.0005
    });
    await this.model.load('models/intraday-model.pt');
  }

  async onBar(bar: any) {
    // Update feature buffer
    this.featureBuffer.push(this.extractFeatures(bar));
    if (this.featureBuffer.length > 60) {
      this.featureBuffer.shift();
    }

    // Need minimum history
    if (this.featureBuffer.length < 60) {
      return;
    }

    // Flatten features for prediction
    const input = this.featureBuffer.flat();

    // Get next-bar prediction
    const prediction = await this.model.predict(input);
    const expectedReturn = prediction.predictions[0];
    const confidence = (prediction.upperBound[0] - prediction.lowerBound[0]);

    // Generate signal if high confidence
    if (Math.abs(expectedReturn) > 0.005 && confidence < 0.01) {
      const signal = {
        symbol: bar.symbol,
        direction: expectedReturn > 0 ? 'long' : 'short',
        confidence: 1 - confidence,
        entryPrice: bar.close,
        reasoning: `Neural model predicts ${(expectedReturn * 100).toFixed(2)}% return`
      };

      // Execute trade
      console.log('Signal generated:', signal);
    }
  }

  private extractFeatures(bar: any): number[] {
    // Extract relevant features
    return [
      (bar.close - bar.open) / bar.open,
      bar.volume / avgVolume,
      // ... more features
    ];
  }
}
```

## API Reference

### NeuralModel

```typescript
class NeuralModel {
  constructor(config: ModelConfig);

  // Train model with data
  train(
    data: number[],
    targets: number[],
    trainingConfig: TrainingConfig
  ): Promise<TrainingMetrics[]>;

  // Make prediction
  predict(inputData: number[]): Promise<PredictionResult>;

  // Save model to file
  save(path: string): Promise<string>;

  // Load model from file
  load(path: string): Promise<void>;

  // Get model information
  getInfo(): Promise<string>;
}
```

### BatchPredictor

```typescript
class BatchPredictor {
  constructor();

  // Add model to batch
  addModel(model: NeuralModel): Promise<number>;

  // Predict on multiple inputs
  predictBatch(inputs: number[][]): Promise<PredictionResult[]>;
}
```

### Functions

```typescript
// List available model types
function listModelTypes(): string[];
```

## Configuration

### ModelConfig Options

| Option | Type | Description | Recommended |
|--------|------|-------------|-------------|
| `modelType` | `string` | Architecture type | 'LSTMAttention' |
| `inputSize` | `number` | Number of input features | 10-100 |
| `horizon` | `number` | Prediction steps ahead | 1-20 |
| `hiddenSize` | `number` | Hidden layer size | 128-512 |
| `numLayers` | `number` | Number of layers | 2-6 |
| `dropout` | `number` | Dropout rate | 0.1-0.3 |
| `learningRate` | `number` | Learning rate | 0.0001-0.001 |

### TrainingConfig Options

| Option | Type | Description | Recommended |
|--------|------|-------------|-------------|
| `epochs` | `number` | Maximum training epochs | 50-200 |
| `batchSize` | `number` | Training batch size | 32-128 |
| `validationSplit` | `number` | Validation data fraction | 0.2 |
| `earlyStoppingPatience` | `number` | Epochs before stopping | 10-20 |
| `useGpu` | `boolean` | Enable GPU acceleration | `true` |

## Examples

### Example 1: Multi-Asset Predictor

```typescript
import { NeuralModel, BatchPredictor } from '@neural-trader/neural';

async function multiAssetForecasting() {
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN'];
  const predictor = new BatchPredictor();

  // Train separate model for each asset
  for (const symbol of symbols) {
    const data = loadHistoricalData(symbol);
    const features = prepareFeatures(data);
    const targets = prepareTargets(data);

    const model = new NeuralModel({
      modelType: 'LSTMAttention',
      inputSize: features[0].length,
      horizon: 5,
      hiddenSize: 128,
      numLayers: 3,
      dropout: 0.2,
      learningRate: 0.001
    });

    await model.train(features, targets, {
      epochs: 100,
      batchSize: 32,
      validationSplit: 0.2,
      earlyStoppingPatience: 10,
      useGpu: true
    });

    await predictor.addModel(model);
  }

  // Make predictions for all assets
  const inputs = symbols.map(s => prepareRecentData(s));
  const predictions = await predictor.predictBatch(inputs);

  predictions.forEach((pred, i) => {
    console.log(`${symbols[i]}: ${(pred.predictions[0] * 100).toFixed(2)}%`);
  });
}
```

### Example 2: Hyperparameter Tuning

```typescript
async function tuneHyperparameters() {
  const configs = [
    { hiddenSize: 128, numLayers: 2, dropout: 0.1 },
    { hiddenSize: 256, numLayers: 3, dropout: 0.2 },
    { hiddenSize: 512, numLayers: 4, dropout: 0.3 }
  ];

  const results = [];

  for (const params of configs) {
    const model = new NeuralModel({
      modelType: 'LSTMAttention',
      inputSize: 50,
      horizon: 5,
      ...params,
      learningRate: 0.001
    });

    const metrics = await model.train(data, targets, trainingConfig);
    const finalMetrics = metrics[metrics.length - 1];

    results.push({
      ...params,
      valLoss: finalMetrics.valLoss,
      valMae: finalMetrics.valMae
    });
  }

  // Find best configuration
  const best = results.sort((a, b) => a.valLoss - b.valLoss)[0];
  console.log('Best Configuration:', best);
}
```

## License

This package is dual-licensed under MIT OR Apache-2.0.

**MIT License**: https://opensource.org/licenses/MIT
**Apache-2.0 License**: https://www.apache.org/licenses/LICENSE-2.0

You may choose either license for your use.
