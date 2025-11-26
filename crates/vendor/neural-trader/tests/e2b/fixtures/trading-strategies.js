/**
 * Trading Strategy Fixtures for E2B Tests
 *
 * Sample trading strategies for testing E2B deployment
 */

/**
 * Simple Moving Average (SMA) Strategy
 */
const smaStrategy = {
  name: 'SMA Strategy',
  code: `
    function calculateSMA(prices, period) {
      if (prices.length < period) return null;
      const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
      return sum / period;
    }

    function generateSignal(prices, shortPeriod = 5, longPeriod = 20) {
      const shortSMA = calculateSMA(prices, shortPeriod);
      const longSMA = calculateSMA(prices, longPeriod);

      if (!shortSMA || !longSMA) return 'HOLD';

      if (shortSMA > longSMA) return 'BUY';
      if (shortSMA < longSMA) return 'SELL';
      return 'HOLD';
    }

    module.exports = { calculateSMA, generateSignal };
  `,
  testData: {
    prices: [100, 102, 101, 105, 103, 107, 106, 110, 108, 112],
    expectedSignal: 'BUY',
  },
};

/**
 * Momentum Strategy
 */
const momentumStrategy = {
  name: 'Momentum Strategy',
  code: `
    function calculateMomentum(prices, period = 5) {
      if (prices.length < period + 1) return null;
      const current = prices[prices.length - 1];
      const previous = prices[prices.length - period - 1];
      return ((current - previous) / previous) * 100;
    }

    function generateSignal(prices, threshold = 2) {
      const momentum = calculateMomentum(prices);
      if (!momentum) return 'HOLD';

      if (momentum > threshold) return 'BUY';
      if (momentum < -threshold) return 'SELL';
      return 'HOLD';
    }

    module.exports = { calculateMomentum, generateSignal };
  `,
  testData: {
    prices: [100, 98, 96, 94, 92, 90, 88, 86, 84, 82],
    expectedSignal: 'SELL',
  },
};

/**
 * Mean Reversion Strategy
 */
const meanReversionStrategy = {
  name: 'Mean Reversion Strategy',
  code: `
    function calculateStats(prices) {
      const mean = prices.reduce((a, b) => a + b) / prices.length;
      const squaredDiffs = prices.map(p => Math.pow(p - mean, 2));
      const variance = squaredDiffs.reduce((a, b) => a + b) / prices.length;
      const stdDev = Math.sqrt(variance);
      return { mean, stdDev };
    }

    function generateSignal(prices, threshold = 2) {
      const { mean, stdDev } = calculateStats(prices);
      const current = prices[prices.length - 1];
      const zScore = (current - mean) / stdDev;

      if (zScore < -threshold) return 'BUY';  // Oversold
      if (zScore > threshold) return 'SELL';  // Overbought
      return 'HOLD';
    }

    module.exports = { calculateStats, generateSignal };
  `,
  testData: {
    prices: [100, 101, 100, 102, 101, 103, 102, 95, 94, 93],
    expectedSignal: 'BUY',
  },
};

/**
 * Bollinger Bands Strategy (Python)
 */
const bollingerBandsStrategyPython = {
  name: 'Bollinger Bands Strategy',
  language: 'python',
  code: `
import json
import math

def calculate_bollinger_bands(prices, period=20, num_std=2):
    if len(prices) < period:
        return None

    sma = sum(prices[-period:]) / period
    squared_diffs = [(p - sma) ** 2 for p in prices[-period:]]
    std_dev = math.sqrt(sum(squared_diffs) / period)

    upper_band = sma + (num_std * std_dev)
    lower_band = sma - (num_std * std_dev)

    return {
        'sma': sma,
        'upper': upper_band,
        'lower': lower_band
    }

def generate_signal(prices):
    bands = calculate_bollinger_bands(prices)
    if not bands:
        return 'HOLD'

    current = prices[-1]

    if current < bands['lower']:
        return 'BUY'
    elif current > bands['upper']:
        return 'SELL'
    else:
        return 'HOLD'

if __name__ == '__main__':
    test_prices = [100, 102, 101, 105, 103, 107, 106, 110, 108, 112]
    signal = generate_signal(test_prices)
    print(json.dumps({'signal': signal}))
  `,
  testData: {
    prices: [100, 102, 101, 105, 103, 107, 106, 110, 108, 112],
    expectedSignal: 'HOLD',
  },
};

/**
 * ML-based Strategy Template (Python)
 */
const mlStrategyTemplate = {
  name: 'ML Strategy Template',
  language: 'python',
  code: `
import json
import random

class SimplePredictor:
    def __init__(self, lookback=5):
        self.lookback = lookback

    def predict(self, prices):
        if len(prices) < self.lookback:
            return None

        recent = prices[-self.lookback:]
        trend = sum([recent[i] - recent[i-1] for i in range(1, len(recent))]) / (len(recent) - 1)

        prediction = prices[-1] + trend
        confidence = min(abs(trend) / 5, 1.0)

        return {
            'predicted_price': round(prediction, 2),
            'current_price': prices[-1],
            'confidence': round(confidence, 2),
            'trend': 'bullish' if trend > 0 else 'bearish'
        }

def generate_signal(prices):
    predictor = SimplePredictor()
    prediction = predictor.predict(prices)

    if not prediction:
        return 'HOLD'

    if prediction['trend'] == 'bullish' and prediction['confidence'] > 0.6:
        return 'BUY'
    elif prediction['trend'] == 'bearish' and prediction['confidence'] > 0.6:
        return 'SELL'
    else:
        return 'HOLD'

if __name__ == '__main__':
    test_prices = [100, 102, 104, 106, 108, 110, 112]
    signal = generate_signal(test_prices)
    print(json.dumps({'signal': signal}))
  `,
  testData: {
    prices: [100, 102, 104, 106, 108, 110, 112],
    expectedSignal: 'BUY',
  },
};

/**
 * Get all strategies
 */
function getAllStrategies() {
  return [
    smaStrategy,
    momentumStrategy,
    meanReversionStrategy,
    bollingerBandsStrategyPython,
    mlStrategyTemplate,
  ];
}

/**
 * Get strategy by name
 */
function getStrategyByName(name) {
  return getAllStrategies().find(s => s.name === name);
}

/**
 * Get strategies by language
 */
function getStrategiesByLanguage(language) {
  return getAllStrategies().filter(s => s.language === language || !s.language);
}

module.exports = {
  smaStrategy,
  momentumStrategy,
  meanReversionStrategy,
  bollingerBandsStrategyPython,
  mlStrategyTemplate,
  getAllStrategies,
  getStrategyByName,
  getStrategiesByLanguage,
};
