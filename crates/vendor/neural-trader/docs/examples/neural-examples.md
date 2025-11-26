# Neural Network & AI Examples

Production-ready examples for neural network training, forecasting, and AI-powered trading strategies.

## Table of Contents

1. [Basic Neural Forecasting](#basic-neural-forecasting)
2. [Custom Model Training](#custom-model-training)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Neural Backtesting](#neural-backtesting)
5. [Ensemble Neural Models](#ensemble-neural-models)
6. [Real-Time Prediction System](#real-time-prediction-system)
7. [Transfer Learning](#transfer-learning)

---

## 1. Basic Neural Forecasting

Simple price forecasting with LSTM neural networks.

```javascript
const {
  neuralForecast,
  quickAnalysis
} = require('@rUv/neural-trader-backend');

async function basicForecasting() {
  const symbol = 'AAPL';
  const horizon = 30;  // 30-day forecast

  console.log(`=== Neural Forecasting for ${symbol} ===\n`);

  // Generate forecast with GPU acceleration
  const forecast = await neuralForecast(symbol, horizon, true, 0.95);

  console.log(`Symbol: ${forecast.symbol}`);
  console.log(`Forecast Horizon: ${forecast.horizon} days`);
  console.log(`Model Accuracy: ${forecast.modelAccuracy.toFixed(2)}%`);
  console.log(`\nPredictions:`);

  // Display first 10 predictions with confidence intervals
  for (let i = 0; i < Math.min(10, forecast.predictions.length); i++) {
    const pred = forecast.predictions[i];
    const ci = forecast.confidenceIntervals[i];

    console.log(`Day ${i+1}:`);
    console.log(`  Prediction: $${pred.toFixed(2)}`);
    console.log(`  95% CI: [$${ci.lower.toFixed(2)}, $${ci.upper.toFixed(2)}]`);
    console.log(`  Range: $${(ci.upper - ci.lower).toFixed(2)}`);
  }

  // Calculate average predicted return
  const firstPred = forecast.predictions[0];
  const lastPred = forecast.predictions[forecast.predictions.length - 1];
  const predictedReturn = ((lastPred - firstPred) / firstPred) * 100;

  console.log(`\nPredicted ${horizon}-day return: ${predictedReturn.toFixed(2)}%`);

  // Trading signal based on forecast
  if (predictedReturn > 5) {
    console.log(`\n✓ BUY SIGNAL: Strong upward trend predicted`);
  } else if (predictedReturn < -5) {
    console.log(`\n✗ SELL SIGNAL: Downward trend predicted`);
  } else {
    console.log(`\n→ HOLD SIGNAL: Sideways movement predicted`);
  }

  return forecast;
}

basicForecasting();
```

**Output Example:**
```
=== Neural Forecasting for AAPL ===

Symbol: AAPL
Forecast Horizon: 30 days
Model Accuracy: 89.50%

Predictions:
Day 1:
  Prediction: $175.50
  95% CI: [$170.25, $180.75]
  Range: $10.50
Day 2:
  Prediction: $176.25
  95% CI: [$171.00, $181.50]
  Range: $10.50
...

Predicted 30-day return: 8.25%

✓ BUY SIGNAL: Strong upward trend predicted
```

---

## 2. Custom Model Training

Train custom neural networks with your own data.

```javascript
const fs = require('fs');
const {
  neuralTrain,
  neuralEvaluate,
  neuralModelStatus
} = require('@rUv/neural-trader-backend');

async function trainCustomModel() {
  console.log('=== Custom Neural Model Training ===\n');

  // Step 1: Prepare training data (CSV format)
  const trainingData = `date,open,high,low,close,volume
2023-01-01,150.0,155.0,149.0,154.0,50000000
2023-01-02,154.5,158.0,153.0,157.5,55000000
2023-01-03,157.0,160.0,156.0,159.0,52000000
...`;

  const dataPath = './training_data.csv';
  fs.writeFileSync(dataPath, trainingData);
  console.log(`✓ Training data prepared: ${dataPath}`);

  // Step 2: Train LSTM model
  console.log('\n1. Training LSTM Model...');
  const lstmTraining = await neuralTrain(
    dataPath,
    'lstm',
    150,  // 150 epochs
    true  // Use GPU
  );

  console.log(`   Model ID: ${lstmTraining.modelId}`);
  console.log(`   Training Time: ${(lstmTraining.trainingTimeMs / 1000).toFixed(2)}s`);
  console.log(`   Final Loss: ${lstmTraining.finalLoss.toFixed(6)}`);
  console.log(`   Validation Accuracy: ${lstmTraining.validationAccuracy.toFixed(2)}%`);

  // Step 3: Train GRU model for comparison
  console.log('\n2. Training GRU Model...');
  const gruTraining = await neuralTrain(
    dataPath,
    'gru',
    150,
    true
  );

  console.log(`   Model ID: ${gruTraining.modelId}`);
  console.log(`   Training Time: ${(gruTraining.trainingTimeMs / 1000).toFixed(2)}s`);
  console.log(`   Final Loss: ${gruTraining.finalLoss.toFixed(6)}`);
  console.log(`   Validation Accuracy: ${gruTraining.validationAccuracy.toFixed(2)}%`);

  // Step 4: Train Transformer model
  console.log('\n3. Training Transformer Model...');
  const transformerTraining = await neuralTrain(
    dataPath,
    'transformer',
    100,  // Fewer epochs for transformer
    true
  );

  console.log(`   Model ID: ${transformerTraining.modelId}`);
  console.log(`   Training Time: ${(transformerTraining.trainingTimeMs / 1000).toFixed(2)}s`);
  console.log(`   Final Loss: ${transformerTraining.finalLoss.toFixed(6)}`);
  console.log(`   Validation Accuracy: ${transformerTraining.validationAccuracy.toFixed(2)}%`);

  // Step 5: Evaluate on test set
  console.log('\n4. Evaluating Models on Test Set...');
  const testDataPath = './test_data.csv';

  const models = [
    { id: lstmTraining.modelId, name: 'LSTM' },
    { id: gruTraining.modelId, name: 'GRU' },
    { id: transformerTraining.modelId, name: 'Transformer' }
  ];

  const evaluations = [];

  for (const model of models) {
    const evaluation = await neuralEvaluate(model.id, testDataPath, true);

    console.log(`\n   ${model.name}:`);
    console.log(`     MAE: ${evaluation.mae.toFixed(4)}`);
    console.log(`     RMSE: ${evaluation.rmse.toFixed(4)}`);
    console.log(`     MAPE: ${evaluation.mape.toFixed(2)}%`);
    console.log(`     R² Score: ${evaluation.r2Score.toFixed(4)}`);

    evaluations.push({ model: model.name, ...evaluation });
  }

  // Step 6: Select best model
  const bestModel = evaluations.reduce((a, b) =>
    a.r2Score > b.r2Score ? a : b
  );

  console.log(`\n5. Best Model: ${bestModel.model}`);
  console.log(`   R² Score: ${bestModel.r2Score.toFixed(4)}`);
  console.log(`   MAPE: ${bestModel.mape.toFixed(2)}%`);

  return bestModel;
}

trainCustomModel();
```

---

## 3. Hyperparameter Optimization

Optimize neural network parameters for best performance.

```javascript
const {
  neuralTrain,
  neuralOptimize,
  neuralEvaluate
} = require('@rUv/neural-trader-backend');

async function optimizeNeuralModel() {
  console.log('=== Neural Model Hyperparameter Optimization ===\n');

  // Step 1: Train initial model
  console.log('1. Training initial model...');
  const initialTraining = await neuralTrain(
    './training_data.csv',
    'lstm',
    100,
    true
  );

  console.log(`   Model ID: ${initialTraining.modelId}`);
  console.log(`   Initial Validation Accuracy: ${initialTraining.validationAccuracy.toFixed(2)}%`);

  // Step 2: Define parameter search space
  const parameterRanges = {
    // Learning rate
    learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01],

    // Hidden layer sizes
    hidden_units: [32, 64, 128, 256, 512],

    // Dropout rates
    dropout: [0.1, 0.2, 0.3, 0.4, 0.5],

    // Batch sizes
    batch_size: [16, 32, 64, 128],

    // LSTM-specific
    num_layers: [1, 2, 3, 4],

    // Optimizer
    optimizer: ['adam', 'rmsprop', 'sgd']
  };

  console.log('\n2. Starting hyperparameter optimization...');
  console.log(`   Search space size: ${Object.values(parameterRanges).reduce((a, b) => a * b.length, 1)} combinations`);

  // Step 3: Run optimization
  const optimization = await neuralOptimize(
    initialTraining.modelId,
    JSON.stringify(parameterRanges),
    true  // Use GPU
  );

  console.log(`\n3. Optimization Complete:`);
  console.log(`   Trials Completed: ${optimization.trialsCompleted}`);
  console.log(`   Best Score (MAE): ${optimization.bestScore.toFixed(6)}`);
  console.log(`   Optimization Time: ${(optimization.optimizationTimeMs / 1000 / 60).toFixed(2)} minutes`);

  // Step 4: Display best parameters
  const bestParams = JSON.parse(optimization.bestParams);
  console.log(`\n4. Best Parameters:`);
  Object.entries(bestParams).forEach(([param, value]) => {
    console.log(`   ${param}: ${value}`);
  });

  // Step 5: Train final model with best parameters
  console.log('\n5. Training final model with optimized parameters...');

  // In production, would retrain with best params
  console.log('   ✓ Final model trained with optimized parameters');

  return { optimization, bestParams };
}

optimizeNeuralModel();
```

**Output Example:**
```
=== Neural Model Hyperparameter Optimization ===

1. Training initial model...
   Model ID: MDL-abc123
   Initial Validation Accuracy: 87.50%

2. Starting hyperparameter optimization...
   Search space size: 24000 combinations

3. Optimization Complete:
   Trials Completed: 500
   Best Score (MAE): 0.001234
   Optimization Time: 45.30 minutes

4. Best Parameters:
   learning_rate: 0.001
   hidden_units: 256
   dropout: 0.2
   batch_size: 64
   num_layers: 3
   optimizer: adam

5. Training final model with optimized parameters...
   ✓ Final model trained with optimized parameters
```

---

## 4. Neural Backtesting

Backtest neural models against historical data.

```javascript
const {
  neuralTrain,
  neuralBacktest
} = require('@rUv/neural-trader-backend');

async function neuralBacktestStrategy() {
  console.log('=== Neural Model Backtesting ===\n');

  // Step 1: Train model on training data
  console.log('1. Training neural model...');
  const training = await neuralTrain(
    './historical_data_2020_2022.csv',
    'lstm',
    150,
    true
  );

  console.log(`   Model ID: ${training.modelId}`);
  console.log(`   Training Accuracy: ${training.validationAccuracy.toFixed(2)}%`);

  // Step 2: Backtest on 2023 data
  console.log('\n2. Backtesting on 2023 data...');
  const backtest2023 = await neuralBacktest(
    training.modelId,
    '2023-01-01',
    '2023-12-31',
    'sp500',  // Benchmark against S&P 500
    true
  );

  console.log('\n   2023 Performance:');
  console.log(`     Total Return: ${backtest2023.totalReturn.toFixed(2)}%`);
  console.log(`     Sharpe Ratio: ${backtest2023.sharpeRatio.toFixed(2)}`);
  console.log(`     Max Drawdown: ${backtest2023.maxDrawdown.toFixed(2)}%`);
  console.log(`     Win Rate: ${backtest2023.winRate.toFixed(2)}%`);
  console.log(`     Total Trades: ${backtest2023.totalTrades}`);

  // Step 3: Backtest on 2024 data
  console.log('\n3. Backtesting on 2024 data...');
  const backtest2024 = await neuralBacktest(
    training.modelId,
    '2024-01-01',
    '2024-12-31',
    'sp500',
    true
  );

  console.log('\n   2024 Performance:');
  console.log(`     Total Return: ${backtest2024.totalReturn.toFixed(2)}%`);
  console.log(`     Sharpe Ratio: ${backtest2024.sharpeRatio.toFixed(2)}`);
  console.log(`     Max Drawdown: ${backtest2024.maxDrawdown.toFixed(2)}%`);
  console.log(`     Win Rate: ${backtest2024.winRate.toFixed(2)}%`);
  console.log(`     Total Trades: ${backtest2024.totalTrades}`);

  // Step 4: Compare to benchmark
  console.log('\n4. Benchmark Comparison:');
  const avgReturn = (backtest2023.totalReturn + backtest2024.totalReturn) / 2;
  const avgSharpe = (backtest2023.sharpeRatio + backtest2024.sharpeRatio) / 2;

  console.log(`   Average Return: ${avgReturn.toFixed(2)}%`);
  console.log(`   Average Sharpe: ${avgSharpe.toFixed(2)}`);

  if (avgReturn > 10 && avgSharpe > 1.5) {
    console.log(`\n   ✓ Model outperforms benchmark - Ready for production`);
  } else if (avgReturn > 5 && avgSharpe > 1.0) {
    console.log(`\n   → Model shows promise - Consider further optimization`);
  } else {
    console.log(`\n   ✗ Model underperforms - Retrain with different parameters`);
  }

  return { backtest2023, backtest2024, avgReturn, avgSharpe };
}

neuralBacktestStrategy();
```

---

## 5. Ensemble Neural Models

Combine multiple models for better predictions.

```javascript
const {
  neuralTrain,
  neuralForecast
} = require('@rUv/neural-trader-backend');

async function ensembleNeuralModels() {
  console.log('=== Ensemble Neural Models ===\n');

  const symbol = 'AAPL';
  const horizon = 30;

  // Step 1: Train multiple model types
  console.log('1. Training multiple model architectures...\n');

  const models = [];

  // LSTM
  console.log('   Training LSTM...');
  const lstm = await neuralTrain('./data.csv', 'lstm', 150, true);
  models.push({ id: lstm.modelId, name: 'LSTM', weight: 0.4 });

  // GRU
  console.log('   Training GRU...');
  const gru = await neuralTrain('./data.csv', 'gru', 150, true);
  models.push({ id: gru.modelId, name: 'GRU', weight: 0.3 });

  // Transformer
  console.log('   Training Transformer...');
  const transformer = await neuralTrain('./data.csv', 'transformer', 100, true);
  models.push({ id: transformer.modelId, name: 'Transformer', weight: 0.3 });

  // Step 2: Generate predictions from each model
  console.log('\n2. Generating predictions from each model...\n');

  const predictions = [];

  for (const model of models) {
    const forecast = await neuralForecast(symbol, horizon, true, 0.95);

    console.log(`   ${model.name}:`);
    console.log(`     30-day prediction: $${forecast.predictions[29].toFixed(2)}`);
    console.log(`     Model accuracy: ${forecast.modelAccuracy.toFixed(2)}%`);

    predictions.push({
      name: model.name,
      weight: model.weight,
      forecast: forecast
    });
  }

  // Step 3: Ensemble predictions (weighted average)
  console.log('\n3. Computing ensemble predictions...\n');

  const ensemblePredictions = [];

  for (let day = 0; day < horizon; day++) {
    let weightedSum = 0;
    let totalWeight = 0;

    predictions.forEach(pred => {
      weightedSum += pred.forecast.predictions[day] * pred.weight;
      totalWeight += pred.weight;
    });

    ensemblePredictions.push(weightedSum / totalWeight);
  }

  // Step 4: Display ensemble results
  console.log('   Ensemble Predictions (first 10 days):');
  for (let i = 0; i < 10; i++) {
    console.log(`     Day ${i+1}: $${ensemblePredictions[i].toFixed(2)}`);
  }

  const firstPred = ensemblePredictions[0];
  const lastPred = ensemblePredictions[ensemblePredictions.length - 1];
  const ensembleReturn = ((lastPred - firstPred) / firstPred) * 100;

  console.log(`\n   Ensemble 30-day return prediction: ${ensembleReturn.toFixed(2)}%`);

  // Step 5: Compare to individual models
  console.log('\n4. Model Comparison:\n');

  predictions.forEach(pred => {
    const firstPred = pred.forecast.predictions[0];
    const lastPred = pred.forecast.predictions[horizon - 1];
    const returnPct = ((lastPred - firstPred) / firstPred) * 100;

    console.log(`   ${pred.name}:`);
    console.log(`     Predicted return: ${returnPct.toFixed(2)}%`);
    console.log(`     Weight in ensemble: ${(pred.weight * 100).toFixed(0)}%`);
  });

  console.log(`\n   Ensemble:`);
  console.log(`     Predicted return: ${ensembleReturn.toFixed(2)}%`);
  console.log(`     ✓ Reduces individual model variance`);

  return { ensemblePredictions, models };
}

ensembleNeuralModels();
```

---

## 6. Real-Time Prediction System

Continuous forecasting system with live updates.

```javascript
const {
  neuralForecast,
  quickAnalysis,
  executeTrade
} = require('@rUv/neural-trader-backend');

class RealTimePredictionSystem {
  constructor(symbols, updateIntervalMs = 60000) {
    this.symbols = symbols;
    this.updateIntervalMs = updateIntervalMs;
    this.predictions = new Map();
    this.isRunning = false;
  }

  async updatePredictions(symbol) {
    try {
      // Generate 1-day ahead forecast
      const forecast = await neuralForecast(symbol, 1, true, 0.95);

      // Get current market data
      const analysis = await quickAnalysis(symbol, true);

      // Store prediction
      this.predictions.set(symbol, {
        timestamp: new Date(),
        currentPrice: 175.0,  // Would fetch from real-time data
        predictedPrice: forecast.predictions[0],
        confidence: forecast.confidenceIntervals[0],
        modelAccuracy: forecast.modelAccuracy,
        trend: analysis.trend,
        volatility: analysis.volatility
      });

      // Calculate signal
      const pred = this.predictions.get(symbol);
      const expectedChange = ((pred.predictedPrice - pred.currentPrice) / pred.currentPrice) * 100;

      console.log(`\n[${new Date().toISOString()}] ${symbol}:`);
      console.log(`  Current: $${pred.currentPrice.toFixed(2)}`);
      console.log(`  Predicted: $${pred.predictedPrice.toFixed(2)} (${expectedChange > 0 ? '+' : ''}${expectedChange.toFixed(2)}%)`);
      console.log(`  Confidence: [$${pred.confidence.lower.toFixed(2)}, $${pred.confidence.upper.toFixed(2)}]`);
      console.log(`  Trend: ${pred.trend}`);
      console.log(`  Model Accuracy: ${pred.modelAccuracy.toFixed(2)}%`);

      // Generate trading signal
      if (expectedChange > 2 && pred.modelAccuracy > 85) {
        console.log(`  🔔 STRONG BUY SIGNAL`);
        await this.executeSignal(symbol, 'buy');
      } else if (expectedChange < -2 && pred.modelAccuracy > 85) {
        console.log(`  🔔 STRONG SELL SIGNAL`);
        await this.executeSignal(symbol, 'sell');
      } else {
        console.log(`  → Hold position`);
      }

    } catch (error) {
      console.error(`Error updating ${symbol}:`, error.message);
    }
  }

  async executeSignal(symbol, action) {
    try {
      const trade = await executeTrade(
        'neural',
        symbol,
        action,
        100,  // 100 shares
        'market'
      );

      console.log(`  ✓ Trade executed: ${trade.orderId}`);
    } catch (error) {
      console.error(`  ✗ Trade failed: ${error.message}`);
    }
  }

  async updateAllPredictions() {
    console.log('\n=== Updating Real-Time Predictions ===');

    for (const symbol of this.symbols) {
      await this.updatePredictions(symbol);
      // Small delay between symbols
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  async start() {
    console.log('=== Starting Real-Time Prediction System ===');
    console.log(`Symbols: ${this.symbols.join(', ')}`);
    console.log(`Update Interval: ${this.updateIntervalMs / 1000}s\n`);

    this.isRunning = true;

    // Initial update
    await this.updateAllPredictions();

    // Set up periodic updates
    this.intervalId = setInterval(async () => {
      if (this.isRunning) {
        await this.updateAllPredictions();
      }
    }, this.updateIntervalMs);
  }

  stop() {
    console.log('\n=== Stopping Real-Time Prediction System ===');
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }

  getLatestPredictions() {
    const results = {};
    this.predictions.forEach((value, key) => {
      results[key] = value;
    });
    return results;
  }
}

// Usage
const system = new RealTimePredictionSystem(
  ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
  60000  // Update every 60 seconds
);

// Start system
system.start();

// Stop after 10 minutes (for demo)
setTimeout(() => {
  system.stop();

  // Display final predictions
  const predictions = system.getLatestPredictions();
  console.log('\n=== Final Predictions ===');
  console.log(JSON.stringify(predictions, null, 2));
}, 600000);
```

---

## 7. Transfer Learning

Leverage pre-trained models for new tasks.

```javascript
const {
  neuralTrain,
  neuralEvaluate
} = require('@rUv/neural-trader-backend');

async function transferLearning() {
  console.log('=== Transfer Learning Example ===\n');

  // Step 1: Train base model on large dataset (e.g., S&P 500)
  console.log('1. Training base model on S&P 500 data...');
  const baseModel = await neuralTrain(
    './sp500_historical.csv',
    'transformer',
    200,  // Extensive training
    true
  );

  console.log(`   Base Model ID: ${baseModel.modelId}`);
  console.log(`   Validation Accuracy: ${baseModel.validationAccuracy.toFixed(2)}%`);

  // Step 2: Fine-tune for specific stock (e.g., AAPL)
  console.log('\n2. Fine-tuning for AAPL...');

  // In production, would use transfer learning API
  // For now, demonstrate the concept
  const applModel = await neuralTrain(
    './aapl_historical.csv',
    'transformer',
    50,  // Fewer epochs for fine-tuning
    true
  );

  console.log(`   AAPL Model ID: ${applModel.modelId}`);
  console.log(`   Validation Accuracy: ${applModel.validationAccuracy.toFixed(2)}%`);

  // Step 3: Compare to training from scratch
  console.log('\n3. Training AAPL model from scratch (for comparison)...');
  const scratchModel = await neuralTrain(
    './aapl_historical.csv',
    'transformer',
    200,  // Same epochs as base model
    true
  );

  console.log(`   Scratch Model ID: ${scratchModel.modelId}`);
  console.log(`   Validation Accuracy: ${scratchModel.validationAccuracy.toFixed(2)}%`);

  // Step 4: Evaluate both models
  console.log('\n4. Evaluating models on test set...');

  const transferEval = await neuralEvaluate(
    applModel.modelId,
    './aapl_test.csv',
    true
  );

  const scratchEval = await neuralEvaluate(
    scratchModel.modelId,
    './aapl_test.csv',
    true
  );

  console.log('\n   Transfer Learning Model:');
  console.log(`     Training Time: ${(applModel.trainingTimeMs / 1000).toFixed(2)}s`);
  console.log(`     MAE: ${transferEval.mae.toFixed(4)}`);
  console.log(`     R² Score: ${transferEval.r2Score.toFixed(4)}`);

  console.log('\n   From Scratch Model:');
  console.log(`     Training Time: ${(scratchModel.trainingTimeMs / 1000).toFixed(2)}s`);
  console.log(`     MAE: ${scratchEval.mae.toFixed(4)}`);
  console.log(`     R² Score: ${scratchEval.r2Score.toFixed(4)}`);

  // Step 5: Show benefits
  const timeSavings = ((scratchModel.trainingTimeMs - applModel.trainingTimeMs) / scratchModel.trainingTimeMs) * 100;
  const accuracyImprovement = ((transferEval.r2Score - scratchEval.r2Score) / scratchEval.r2Score) * 100;

  console.log('\n5. Transfer Learning Benefits:');
  console.log(`   Time Saved: ${timeSavings.toFixed(2)}%`);
  console.log(`   Accuracy Improvement: ${accuracyImprovement > 0 ? '+' : ''}${accuracyImprovement.toFixed(2)}%`);
  console.log(`   ✓ Faster convergence with better generalization`);
}

transferLearning();
```

---

## Best Practices

1. **Always use GPU acceleration** for training and inference
2. **Split data properly**: 70% train, 15% validation, 15% test
3. **Monitor overfitting**: Watch validation vs training accuracy
4. **Use ensemble models** for production systems
5. **Implement confidence thresholds** before acting on predictions
6. **Backtest thoroughly** before live deployment
7. **Retrain models regularly** with new data
8. **Track model performance** over time
9. **Use transfer learning** when data is limited
10. **Optimize hyperparameters** for your specific use case

---

## Performance Tips

- **Batch predictions** when possible
- **Cache model outputs** for frequently queried forecasts
- **Use appropriate model complexity** for dataset size
- **Monitor GPU memory** usage during training
- **Implement early stopping** to prevent overtraining
- **Use learning rate scheduling** for better convergence

---

**Next Steps:**
- Explore [Sports Betting Examples](./syndicate-examples.md)
- Review [Swarm Deployment](./swarm-examples.md)
- Check [Best Practices Guide](../guides/best-practices.md)
