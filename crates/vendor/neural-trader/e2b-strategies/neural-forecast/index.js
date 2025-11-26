/**
 * Neural Forecasting Strategy for E2B Sandbox
 * Symbols: AAPL, TSLA, NVDA
 * Strategy: LSTM-based price prediction with confidence-based position sizing
 */

const Alpaca = require('@alpacahq/alpaca-trade-api');
const express = require('express');
const tf = require('@tensorflow/tfjs-node');

// Configuration
const config = {
  alpacaKey: process.env.ALPACA_API_KEY,
  alpacaSecret: process.env.ALPACA_SECRET_KEY,
  alpacaBaseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
  symbols: ['AAPL', 'TSLA', 'NVDA'],
  confidenceThreshold: 0.70,
  lookbackPeriods: 60,
  forecastHorizon: 5,
  maxPositionSize: 50,
  minPositionSize: 5,
  port: process.env.PORT || 3001
};

const alpaca = new Alpaca({
  keyId: config.alpacaKey,
  secretKey: config.alpacaSecret,
  paper: true,
  baseUrl: config.alpacaBaseUrl
});

const logger = {
  info: (msg, data = {}) => console.log(JSON.stringify({ level: 'INFO', msg, ...data, timestamp: new Date().toISOString() })),
  error: (msg, data = {}) => console.error(JSON.stringify({ level: 'ERROR', msg, ...data, timestamp: new Date().toISOString() })),
  trade: (msg, data = {}) => console.log(JSON.stringify({ level: 'TRADE', msg, ...data, timestamp: new Date().toISOString() })),
  model: (msg, data = {}) => console.log(JSON.stringify({ level: 'MODEL', msg, ...data, timestamp: new Date().toISOString() }))
};

// Store models and training data
const models = new Map();
const trainingData = new Map();

/**
 * Normalize price data for neural network
 */
function normalizeData(data) {
  const prices = data.map(d => d.ClosePrice);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min;

  return {
    normalized: prices.map(p => (p - min) / range),
    min,
    max,
    range
  };
}

/**
 * Create LSTM model for price prediction
 */
function createLSTMModel() {
  const model = tf.sequential();

  // LSTM layer
  model.add(tf.layers.lstm({
    units: 50,
    returnSequences: true,
    inputShape: [config.lookbackPeriods, 1]
  }));

  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Second LSTM layer
  model.add(tf.layers.lstm({
    units: 50,
    returnSequences: false
  }));

  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Dense layers
  model.add(tf.layers.dense({ units: 25, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  return model;
}

/**
 * Prepare training data from price bars
 */
function prepareTrainingData(normalizedPrices) {
  const X = [];
  const y = [];

  for (let i = 0; i < normalizedPrices.length - config.lookbackPeriods - config.forecastHorizon; i++) {
    X.push(normalizedPrices.slice(i, i + config.lookbackPeriods));
    y.push(normalizedPrices[i + config.lookbackPeriods + config.forecastHorizon - 1]);
  }

  return { X, y };
}

/**
 * Train or retrain model for a symbol
 */
async function trainModel(symbol, bars) {
  try {
    logger.model('Training model', { symbol, barCount: bars.length });

    const { normalized, min, max, range } = normalizeData(bars);
    const { X, y } = prepareTrainingData(normalized);

    if (X.length < 10) {
      logger.error('Insufficient training data', { symbol, sampleCount: X.length });
      return null;
    }

    const model = createLSTMModel();

    const xs = tf.tensor3d(X.map(seq => seq.map(val => [val])));
    const ys = tf.tensor2d(y.map(val => [val]));

    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 10 === 0) {
            logger.model('Training progress', { symbol, epoch, loss: logs.loss.toFixed(4) });
          }
        }
      }
    });

    xs.dispose();
    ys.dispose();

    models.set(symbol, { model, min, max, range });
    logger.model('Model trained successfully', { symbol });

    return model;

  } catch (error) {
    logger.error('Model training error', { symbol, error: error.message });
    return null;
  }
}

/**
 * Make price prediction with confidence score
 */
async function predict(symbol, recentBars) {
  try {
    const modelData = models.get(symbol);
    if (!modelData) {
      logger.error('Model not found', { symbol });
      return null;
    }

    const { model, min, max, range } = modelData;

    // Normalize recent prices
    const recentPrices = recentBars.slice(-config.lookbackPeriods).map(b => b.ClosePrice);
    const normalizedRecent = recentPrices.map(p => (p - min) / range);

    // Prepare input tensor
    const inputTensor = tf.tensor3d([normalizedRecent.map(val => [val])]);

    // Predict
    const prediction = model.predict(inputTensor);
    const predictedNormalized = (await prediction.data())[0];

    inputTensor.dispose();
    prediction.dispose();

    // Denormalize prediction
    const predictedPrice = predictedNormalized * range + min;
    const currentPrice = recentBars[recentBars.length - 1].ClosePrice;

    // Calculate confidence based on prediction magnitude and volatility
    const priceChange = Math.abs(predictedPrice - currentPrice) / currentPrice;
    const volatility = calculateVolatility(recentBars);
    const confidence = Math.min(0.95, Math.max(0.5, 1 - (volatility / priceChange)));

    logger.model('Prediction made', {
      symbol,
      currentPrice,
      predictedPrice,
      priceChange: (priceChange * 100).toFixed(2) + '%',
      confidence: (confidence * 100).toFixed(2) + '%'
    });

    return {
      currentPrice,
      predictedPrice,
      priceChange,
      confidence,
      direction: predictedPrice > currentPrice ? 'up' : 'down'
    };

  } catch (error) {
    logger.error('Prediction error', { symbol, error: error.message });
    return null;
  }
}

/**
 * Calculate historical volatility
 */
function calculateVolatility(bars) {
  const returns = [];
  for (let i = 1; i < bars.length; i++) {
    const ret = (bars[i].ClosePrice - bars[i - 1].ClosePrice) / bars[i - 1].ClosePrice;
    returns.push(ret);
  }

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
  return Math.sqrt(variance);
}

/**
 * Calculate position size based on confidence
 */
function calculatePositionSize(confidence) {
  const normalized = (confidence - config.confidenceThreshold) / (1 - config.confidenceThreshold);
  const size = config.minPositionSize + normalized * (config.maxPositionSize - config.minPositionSize);
  return Math.round(size);
}

/**
 * Get historical bars for training and prediction
 */
async function getHistoricalBars(symbol) {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - 7 * 24 * 60 * 60 * 1000); // 7 days

    const bars = await alpaca.getBarsV2(symbol, {
      start: startDate.toISOString(),
      end: endDate.toISOString(),
      timeframe: '5Min',
      limit: 1000
    });

    const barArray = [];
    for await (let bar of bars) {
      barArray.push(bar);
    }

    return barArray;

  } catch (error) {
    logger.error('Error fetching bars', { symbol, error: error.message });
    return null;
  }
}

/**
 * Check current position
 */
async function getPosition(symbol) {
  try {
    const position = await alpaca.getPosition(symbol);
    return parseInt(position.qty);
  } catch (error) {
    if (error.message.includes('position does not exist')) {
      return 0;
    }
    throw error;
  }
}

/**
 * Execute trading logic with neural forecasting
 */
async function tradingLogic(symbol) {
  try {
    const bars = await getHistoricalBars(symbol);
    if (!bars || bars.length < config.lookbackPeriods + 50) {
      logger.error('Insufficient historical data', { symbol });
      return;
    }

    // Train/retrain model if not exists or periodically
    if (!models.has(symbol) || Math.random() < 0.1) { // 10% chance to retrain
      await trainModel(symbol, bars);
    }

    // Make prediction
    const prediction = await predict(symbol, bars);
    if (!prediction) return;

    const currentPosition = await getPosition(symbol);

    // Trading decisions based on prediction and confidence
    if (prediction.confidence >= config.confidenceThreshold) {
      const positionSize = calculatePositionSize(prediction.confidence);

      if (prediction.direction === 'up' && currentPosition === 0) {
        // Buy signal
        const order = await alpaca.createOrder({
          symbol: symbol,
          qty: positionSize,
          side: 'buy',
          type: 'market',
          time_in_force: 'day'
        });

        logger.trade('BUY order placed', {
          symbol,
          prediction,
          qty: positionSize,
          orderId: order.id
        });
      }
      else if (prediction.direction === 'down' && currentPosition > 0) {
        // Sell signal
        const order = await alpaca.createOrder({
          symbol: symbol,
          qty: currentPosition,
          side: 'sell',
          type: 'market',
          time_in_force: 'day'
        });

        logger.trade('SELL order placed', {
          symbol,
          prediction,
          qty: currentPosition,
          orderId: order.id
        });
      }
    } else {
      logger.info('Confidence below threshold', { symbol, confidence: prediction.confidence });
    }

  } catch (error) {
    logger.error('Trading logic error', { symbol, error: error.message, stack: error.stack });
  }
}

/**
 * Main strategy loop
 */
async function runStrategy() {
  logger.info('Neural forecasting strategy started', { symbols: config.symbols });

  const clock = await alpaca.getClock();
  if (!clock.is_open) {
    logger.info('Market is closed', { nextOpen: clock.next_open });
    return;
  }

  await Promise.all(config.symbols.map(symbol => tradingLogic(symbol)));

  logger.info('Strategy cycle completed');
}

/**
 * Express server
 */
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    strategy: 'neural-forecast',
    symbols: config.symbols,
    modelsLoaded: Array.from(models.keys()),
    timestamp: new Date().toISOString()
  });
});

app.get('/status', async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    const positions = await alpaca.getPositions();

    res.json({
      account: {
        equity: account.equity,
        cash: account.cash,
        buyingPower: account.buying_power
      },
      positions: positions.map(p => ({
        symbol: p.symbol,
        qty: p.qty,
        currentPrice: p.current_price,
        marketValue: p.market_value,
        unrealizedPL: p.unrealized_pl
      })),
      models: Array.from(models.keys())
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/execute', async (req, res) => {
  try {
    await runStrategy();
    res.json({ success: true, message: 'Strategy executed' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/retrain/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const bars = await getHistoricalBars(symbol);
    await trainModel(symbol, bars);
    res.json({ success: true, symbol, message: 'Model retrained' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(config.port, () => {
  logger.info('Neural forecasting server started', { port: config.port });
});

// Run strategy every 15 minutes
setInterval(runStrategy, 15 * 60 * 1000);

// Initial run
runStrategy().catch(error => {
  logger.error('Initial strategy run failed', { error: error.message });
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  // Dispose TensorFlow resources
  models.forEach(({ model }) => model.dispose());
  process.exit(0);
});
