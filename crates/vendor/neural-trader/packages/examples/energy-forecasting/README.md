# @neural-trader/example-energy-forecasting

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-energy-forecasting.svg)](https://www.npmjs.com/package/@neural-trader/example-energy-forecasting)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-energy-forecasting.svg)](https://www.npmjs.com/package/@neural-trader/example-energy-forecasting)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)]()

Self-learning energy forecasting with conformal prediction and swarm-based ensemble models for solar, wind, demand, and temperature prediction.

## Features

- **Multi-Model Ensemble**: ARIMA, LSTM, Transformer, and Prophet models
- **Conformal Prediction**: Guaranteed coverage prediction intervals with @neural-trader/predictor
- **Swarm-Based Optimization**: Parallel hyperparameter exploration for optimal model selection
- **Adaptive Learning**: Online model updates and horizon-specific model weighting
- **Seasonal Pattern Detection**: Automatic detection and modeling of seasonal components
- **Dual Implementation**: Pure TypeScript with automatic WASM/Native acceleration via @neural-trader/predictor
- **Uncertainty Quantification**: Statistically rigorous prediction intervals with configurable confidence levels

## Supported Domains

- ☀️ **Solar Generation**: Photovoltaic power forecasting with daily cycles
- 💨 **Wind Power**: Wind turbine generation with high variability
- ⚡ **Electricity Demand**: Load forecasting with multi-seasonal patterns
- 🌡️ **Temperature**: Weather prediction with seasonal trends

## Installation

```bash
npm install @neural-trader/example-energy-forecasting
```

## Quick Start

```typescript
import { EnergyForecaster, EnergyDomain, ModelType } from '@neural-trader/example-energy-forecasting';

// Create forecaster
const forecaster = new EnergyForecaster(EnergyDomain.SOLAR, {
  alpha: 0.1,              // 90% confidence intervals
  horizon: 48,             // 48-hour forecast
  seasonalPeriod: 24,      // Daily seasonality
  enableAdaptive: true,
  ensembleConfig: {
    models: [ModelType.ARIMA, ModelType.LSTM, ModelType.PROPHET]
  }
});

// Train with historical data
const historicalData = [
  { timestamp: Date.now(), value: 150 },
  // ... more data points
];
await forecaster.train(historicalData);

// Generate forecast
const forecast = await forecaster.forecast(48);

// Access predictions with confidence intervals
forecast.forecasts.forEach(f => {
  console.log(`Time: ${new Date(f.timestamp).toISOString()}`);
  console.log(`Forecast: ${f.pointForecast}`);
  console.log(`Interval: [${f.interval.lower}, ${f.interval.upper}]`);
  console.log(`Confidence: ${f.confidence * 100}%`);
});
```

## Architecture

### Core Components

#### 1. **EnergyForecaster**
Main forecasting system that orchestrates ensemble models and conformal prediction.

```typescript
const forecaster = new EnergyForecaster(EnergyDomain.SOLAR, config);
await forecaster.train(data);
const forecast = await forecaster.forecast(48);
```

#### 2. **EnsembleSwarm**
Swarm-based hyperparameter optimization and adaptive model weighting.

```typescript
const ensemble = new EnsembleSwarm({
  models: [ModelType.ARIMA, ModelType.LSTM],
  adaptiveLearningRate: 0.05,
  retrainFrequency: 100
});
await ensemble.trainEnsemble(trainingData, validationData);
```

#### 3. **EnergyConformalPredictor**
Wrapper around @neural-trader/predictor for uncertainty quantification.

```typescript
const predictor = new EnergyConformalPredictor(ModelType.ARIMA, {
  alpha: 0.1,
  calibrationSize: 2000
}, true); // adaptive mode

await predictor.initialize();
await predictor.calibrate(historicalData, predictions);
```

### Model Implementations

All models implement the `ForecastingModel` interface:

- **ARIMAModel**: Exponential smoothing with trend
- **LSTMModel**: Recurrent patterns with attention-like weights
- **TransformerModel**: Self-attention mechanism
- **ProphetModel**: Trend + seasonality decomposition

## Examples

Run the included examples:

```bash
# Solar generation forecasting
npm run example:solar

# Wind power forecasting
npm run example:wind

# Electricity demand forecasting
npm run example:demand

# Temperature prediction
npm run example:temperature
```

## Configuration

### ForecasterConfig

```typescript
interface ForecasterConfig {
  alpha?: number;                    // Conformal prediction miscoverage rate (default: 0.1)
  calibrationSize?: number;          // Max calibration samples (default: 2000)
  horizon?: number;                  // Default forecast horizon (default: 24)
  seasonalPeriod?: number;           // Seasonal period in hours (default: 24)
  enableAdaptive?: boolean;          // Enable adaptive conformal prediction (default: true)
  ensembleConfig?: EnsembleConfig;   // Ensemble configuration
  weatherIntegration?: WeatherConfig; // Weather API integration
}
```

### EnsembleConfig

```typescript
interface EnsembleConfig {
  models: ModelType[];                              // Models to include in ensemble
  horizonWeights?: Map<number, Map<ModelType, number>>; // Custom horizon weights
  adaptiveLearningRate?: number;                    // Learning rate for weight updates (default: 0.05)
  retrainFrequency?: number;                        // Retraining frequency (default: 100)
  minCalibrationSamples?: number;                   // Min samples for calibration (default: 50)
}
```

## Conformal Prediction

This package uses [@neural-trader/predictor](../predictor) for statistically rigorous uncertainty quantification:

- **Split Conformal Prediction**: Distribution-free prediction intervals
- **Adaptive Conformal Inference**: Dynamic alpha adjustment for target coverage
- **Multiple Implementations**: Pure TypeScript with automatic WASM/Native acceleration

### Coverage Guarantees

With conformal prediction, the forecaster provides mathematically guaranteed coverage:

```
P(y_true ∈ [lower, upper]) ≥ 1 - α
```

For α = 0.1, at least 90% of true values will fall within the prediction intervals.

## Performance

### Model Selection by Domain

Recommended models for each domain based on empirical performance:

- **Solar**: Prophet, ARIMA (strong daily seasonality)
- **Wind**: LSTM, Transformer (high variability)
- **Demand**: Prophet, LSTM (multi-seasonal patterns)
- **Temperature**: Prophet, Transformer (seasonal trends)

### Computational Complexity

- **Training**: O(n·m·k) where n=samples, m=models, k=hyperparameter candidates
- **Prediction**: O(h·m) where h=horizon, m=models in ensemble
- **Calibration**: O(n log n) for conformal predictor
- **Update**: O(log n) for online learning

### Parallel Execution

The swarm exploration runs hyperparameter search in parallel:

```typescript
// All model types explored concurrently
const explorationPromises = models.map(modelType =>
  ensemble.exploreHyperparameters(modelType, data, validation)
);
await Promise.all(explorationPromises);
```

## Testing

Comprehensive test suite with >90% coverage:

```bash
npm test                 # Run all tests
npm run test:watch      # Watch mode
npm run test:coverage   # Coverage report
```

## API Reference

### Main Classes

- **EnergyForecaster**: Main forecasting system
- **EnsembleSwarm**: Swarm-based ensemble learning
- **EnergyConformalPredictor**: Conformal prediction wrapper
- **ARIMAModel**, **LSTMModel**, **TransformerModel**, **ProphetModel**: Model implementations

### Types

- **TimeSeriesPoint**: Time series data point
- **ForecastResult**: Single forecast with prediction interval
- **MultiStepForecast**: Multi-step forecast results
- **ModelPerformance**: Model performance metrics
- **EnergyDomain**: Energy domain enum
- **ModelType**: Model type enum

## Advanced Usage

### Custom Model Weights

```typescript
const horizonWeights = new Map([
  [1, new Map([[ModelType.LSTM, 0.6], [ModelType.ARIMA, 0.4]])],
  [24, new Map([[ModelType.PROPHET, 0.5], [ModelType.LSTM, 0.5]])],
]);

const forecaster = new EnergyForecaster(domain, {
  ensembleConfig: {
    models: [ModelType.LSTM, ModelType.ARIMA, ModelType.PROPHET],
    horizonWeights
  }
});
```

### Online Learning

```typescript
// Initial training
await forecaster.train(historicalData);

// Generate forecast
const forecast = await forecaster.forecast(24);

// Update with new observations as they arrive
for (const observation of newData) {
  await forecaster.update(observation);
}
```

### Accessing Statistics

```typescript
const stats = forecaster.getStats();

console.log('Domain:', stats.domain);
console.log('Training points:', stats.trainingPoints);
console.log('Models:', stats.ensembleStats.modelCount);
console.log('Seasonal strength:', stats.seasonalPattern?.strength);

// Per-model performance
stats.ensembleStats.performances.forEach(({ model, performance }) => {
  console.log(`${model}: MAPE=${performance.mape}%, RMSE=${performance.rmse}`);
});
```

## Integration with Neural Trader

This package integrates seamlessly with the Neural Trader ecosystem:

- **@neural-trader/predictor**: Conformal prediction core
- **agentdb**: Memory-persistent model performance tracking (optional)
- **claude-flow**: Swarm orchestration (optional)
- **openrouter**: Weather pattern interpretation (optional)

## Contributing

Contributions welcome! Please ensure:

- Tests pass: `npm test`
- Linting passes: `npm run lint`
- Types check: `npm run typecheck`
- Coverage >90%: `npm run test:coverage`

## License

MIT

## See Also

- [@neural-trader/predictor](../predictor) - Conformal prediction core
- [Neural Trader](../../README.md) - Main repository
- [Examples](../examples) - More example packages
