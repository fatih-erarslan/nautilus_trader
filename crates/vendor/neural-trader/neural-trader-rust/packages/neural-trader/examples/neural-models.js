/**
 * Neural Trader - Neural Models Example
 *
 * This example demonstrates:
 * - Training neural network models
 * - Making predictions
 * - Model evaluation
 * - Integration with trading strategies
 */

const {
  NeuralModel,
  LSTMModel,
  TransformerModel,
  BacktestEngine,
  Strategy,
  TechnicalIndicators
} = require('neural-trader');

/**
 * Neural Network Trading Strategy
 */
class NeuralTradingStrategy extends Strategy {
  constructor(model) {
    super('Neural Trading');
    this.model = model;
    this.lookback = 60; // Use 60 bars for prediction
  }

  async onBar(bar, portfolio, context) {
    const prices = context.getPriceHistory(this.lookback + 1);

    if (prices.length < this.lookback + 1) return null;

    // Prepare features for neural network
    const features = this.prepareFeatures(prices, context);

    // Get prediction from neural model
    const prediction = await this.model.predict(features);

    // prediction: { direction: 'up'|'down', confidence: 0-1, expectedReturn: number }

    const position = portfolio.getPosition();

    // Trading logic based on neural prediction
    if (prediction.direction === 'up' && prediction.confidence > 0.7 && !position) {
      return {
        action: 'buy',
        quantity: Math.floor(portfolio.cash * prediction.confidence / bar.close),
        price: bar.close,
        confidence: prediction.confidence
      };
    } else if (prediction.direction === 'down' && position) {
      return {
        action: 'sell',
        quantity: position.quantity,
        price: bar.close,
        confidence: prediction.confidence
      };
    }

    return null;
  }

  prepareFeatures(prices, context) {
    // Create features from price data and technical indicators
    const returns = prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);
    const sma20 = TechnicalIndicators.sma(prices, 20);
    const rsi = TechnicalIndicators.rsi(prices, 14);
    const { upper, middle, lower } = TechnicalIndicators.bollingerBands(prices, 20, 2);

    return {
      prices: prices.slice(-this.lookback),
      returns: returns.slice(-this.lookback),
      sma: sma20,
      rsi: rsi,
      bollingerBands: { upper, middle, lower },
      volume: context.getVolumeHistory(this.lookback)
    };
  }
}

/**
 * Train LSTM model for price prediction
 */
async function trainLSTMModel() {
  console.log('🧠 Training LSTM Model\n');

  const model = new LSTMModel({
    inputSize: 60,      // 60 timesteps
    hiddenSize: 128,    // 128 hidden units
    numLayers: 2,       // 2 LSTM layers
    dropout: 0.2,       // 20% dropout
    outputSize: 1       // Predict next price
  });

  // Prepare training data
  console.log('📊 Preparing training data...');
  const trainingData = await prepareTrainingData({
    symbol: 'AAPL',
    startDate: '2020-01-01',
    endDate: '2023-12-31',
    sequenceLength: 60
  });

  // Train the model
  console.log('🏋️  Training model...');
  const history = await model.train(trainingData, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 10 === 0) {
          console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}`);
        }
      }
    }
  });

  console.log('\n✅ Training complete!');
  console.log(`Final Loss: ${history.loss[history.loss.length - 1].toFixed(4)}`);
  console.log(`Final Val Loss: ${history.val_loss[history.val_loss.length - 1].toFixed(4)}`);

  // Evaluate model
  console.log('\n📈 Evaluating model...');
  const testData = await prepareTrainingData({
    symbol: 'AAPL',
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    sequenceLength: 60
  });

  const metrics = await model.evaluate(testData);
  console.log(`Test MAE: ${metrics.mae.toFixed(4)}`);
  console.log(`Test RMSE: ${metrics.rmse.toFixed(4)}`);
  console.log(`Test R²: ${metrics.r2.toFixed(4)}`);

  // Save model
  await model.save('models/lstm_aapl.model');
  console.log('\n💾 Model saved to models/lstm_aapl.model');

  return model;
}

/**
 * Train Transformer model for advanced predictions
 */
async function trainTransformerModel() {
  console.log('🤖 Training Transformer Model\n');

  const model = new TransformerModel({
    inputSize: 60,
    dModel: 128,          // Model dimension
    nHead: 8,             // Number of attention heads
    numLayers: 4,         // Number of transformer layers
    dropout: 0.1,
    outputSize: 3         // Predict: direction, confidence, magnitude
  });

  console.log('📊 Preparing training data with attention features...');
  const trainingData = await prepareTransformerData({
    symbols: ['AAPL', 'GOOGL', 'MSFT'], // Multi-asset learning
    startDate: '2020-01-01',
    endDate: '2023-12-31',
    sequenceLength: 60
  });

  console.log('🏋️  Training transformer...');
  const history = await model.train(trainingData, {
    epochs: 50,
    batchSize: 64,
    learningRate: 0.001,
    validationSplit: 0.2
  });

  console.log('\n✅ Transformer training complete!');

  await model.save('models/transformer_multi.model');
  console.log('💾 Model saved to models/transformer_multi.model');

  return model;
}

/**
 * Use neural model in backtesting
 */
async function backtestNeuralStrategy() {
  console.log('🔬 Backtesting Neural Strategy\n');

  // Load trained model
  const model = await LSTMModel.load('models/lstm_aapl.model');

  // Create strategy with neural model
  const strategy = new NeuralTradingStrategy(model);

  // Run backtest
  const engine = new BacktestEngine({
    initialCapital: 100000,
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    symbol: 'AAPL'
  });

  const results = await engine.run(strategy);

  // Display results
  console.log('📊 Neural Strategy Results:');
  console.log(`Total Return: ${results.totalReturn.toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${results.sharpeRatio.toFixed(2)}`);
  console.log(`Max Drawdown: ${results.maxDrawdown.toFixed(2)}%`);
  console.log(`Win Rate: ${results.winRate.toFixed(2)}%`);
  console.log(`Total Trades: ${results.totalTrades}`);

  // Compare with buy-and-hold
  const buyHoldReturn = results.buyHoldReturn;
  console.log(`\nBuy & Hold Return: ${buyHoldReturn.toFixed(2)}%`);
  console.log(`Alpha: ${(results.totalReturn - buyHoldReturn).toFixed(2)}%`);

  return results;
}

/**
 * Model comparison and ensemble
 */
async function compareModels() {
  console.log('🏆 Comparing Neural Models\n');

  // Load different models
  const lstmModel = await LSTMModel.load('models/lstm_aapl.model');
  const transformerModel = await TransformerModel.load('models/transformer_multi.model');

  // Test both models
  const testData = await prepareTrainingData({
    symbol: 'AAPL',
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    sequenceLength: 60
  });

  console.log('📊 LSTM Performance:');
  const lstmMetrics = await lstmModel.evaluate(testData);
  console.log(`  MAE: ${lstmMetrics.mae.toFixed(4)}`);
  console.log(`  RMSE: ${lstmMetrics.rmse.toFixed(4)}`);
  console.log(`  R²: ${lstmMetrics.r2.toFixed(4)}`);

  console.log('\n📊 Transformer Performance:');
  const transformerMetrics = await transformerModel.evaluate(testData);
  console.log(`  MAE: ${transformerMetrics.mae.toFixed(4)}`);
  console.log(`  RMSE: ${transformerMetrics.rmse.toFixed(4)}`);
  console.log(`  R²: ${transformerMetrics.r2.toFixed(4)}`);

  // Create ensemble model
  const ensemble = NeuralModel.createEnsemble([lstmModel, transformerModel], {
    weights: [0.5, 0.5] // Equal weighting
  });

  console.log('\n📊 Ensemble Performance:');
  const ensembleMetrics = await ensemble.evaluate(testData);
  console.log(`  MAE: ${ensembleMetrics.mae.toFixed(4)}`);
  console.log(`  RMSE: ${ensembleMetrics.rmse.toFixed(4)}`);
  console.log(`  R²: ${ensembleMetrics.r2.toFixed(4)}`);
}

/**
 * Helper: Prepare training data
 */
async function prepareTrainingData(options) {
  // This would fetch and prepare real market data
  // For demo purposes, returning mock structure
  return {
    features: [], // Array of feature vectors
    labels: [],   // Array of target values
    metadata: options
  };
}

/**
 * Helper: Prepare transformer data
 */
async function prepareTransformerData(options) {
  // Similar to prepareTrainingData but with attention features
  return {
    features: [],
    labels: [],
    attention_masks: [],
    metadata: options
  };
}

/**
 * Get starter code for neural models
 */
function getStarterCode() {
  return `const { NeuralModel, LSTMModel, Strategy } = require('neural-trader');

// Train a simple LSTM model
async function trainModel() {
  const model = new LSTMModel({
    inputSize: 60,
    hiddenSize: 128,
    numLayers: 2,
    outputSize: 1
  });

  const trainingData = await loadTrainingData();

  await model.train(trainingData, {
    epochs: 100,
    batchSize: 32
  });

  await model.save('my_model.model');
  return model;
}

// Use model in strategy
class NeuralStrategy extends Strategy {
  constructor(model) {
    super('Neural Strategy');
    this.model = model;
  }

  async onBar(bar, portfolio, context) {
    const features = prepareFeatures(context);
    const prediction = await this.model.predict(features);

    // Trade based on prediction
    if (prediction.direction === 'up') {
      return { action: 'buy', quantity: 100, price: bar.close };
    }

    return null;
  }
}

trainModel().then(model => {
  console.log('Model trained successfully!');
}).catch(console.error);
`;
}

module.exports = {
  trainLSTMModel,
  trainTransformerModel,
  backtestNeuralStrategy,
  compareModels,
  getStarterCode,
  NeuralTradingStrategy
};

// Run if called directly
if (require.main === module) {
  (async () => {
    await trainLSTMModel();
    await trainTransformerModel();
    await backtestNeuralStrategy();
    await compareModels();
  })().catch(console.error);
}
