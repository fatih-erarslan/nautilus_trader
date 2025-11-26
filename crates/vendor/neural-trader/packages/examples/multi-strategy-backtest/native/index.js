/**
 * NAPI-RS wrapper for performance-critical operations
 * Falls back to pure JS if native module isn't available
 */

let nativeModule = null;

try {
  nativeModule = require('./target/release/index.node');
  console.log('✅ Native NAPI-RS module loaded - using optimized calculations');
} catch (error) {
  console.log('⚠️  Native module not available, using pure JS fallback');
  console.log('   Run "npm run native:build" to compile native module for better performance');
}

// Pure JS fallbacks
const fallbacks = {
  calculateSma(prices, period) {
    if (prices.length < period) return [];
    const sma = [];
    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      const avg = slice.reduce((sum, val) => sum + val, 0) / period;
      sma.push(avg);
    }
    return sma;
  },

  calculateEma(prices, period) {
    if (prices.length === 0) return [];
    const multiplier = 2 / (period + 1);
    const ema = [prices[0]];

    for (let i = 1; i < prices.length; i++) {
      const value = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
      ema.push(value);
    }
    return ema;
  },

  calculateReturns(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
  },

  calculateStdDev(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
  },

  calculateSharpeRatio(returns, riskFreeRate = 0) {
    if (returns.length === 0) return 0;
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const stdDev = this.calculateStdDev(returns);
    if (stdDev === 0) return 0;
    const excessReturn = meanReturn - (riskFreeRate / 252);
    return (excessReturn / stdDev) * Math.sqrt(252);
  },

  calculateMaxDrawdown(equityCurve) {
    if (equityCurve.length === 0) return 0;
    let maxDrawdown = 0;
    let peak = equityCurve[0];

    for (const value of equityCurve) {
      if (value > peak) peak = value;
      const drawdown = (peak - value) / peak;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }
    return maxDrawdown;
  },

  calculatePerformanceMetrics(equityCurve, returns, trades, wins, totalProfit, totalLoss) {
    const totalReturn = equityCurve.length > 0 && equityCurve[0] !== 0
      ? (equityCurve[equityCurve.length - 1] - equityCurve[0]) / equityCurve[0]
      : 0;

    const sharpeRatio = this.calculateSharpeRatio(returns);
    const maxDrawdown = this.calculateMaxDrawdown(equityCurve);
    const winRate = trades > 0 ? wins / trades : 0;

    const avgWin = wins > 0 ? totalProfit / wins : 0;
    const avgLoss = trades > wins ? Math.abs(totalLoss) / (trades - wins) : 0;
    const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;

    const calmarRatio = maxDrawdown !== 0 ? totalReturn / maxDrawdown : 0;

    const downsideReturns = returns.filter(r => r < 0);
    const downsideStd = this.calculateStdDev(downsideReturns);
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const sortinoRatio = downsideStd > 0 ? (meanReturn / downsideStd) * Math.sqrt(252) : 0;

    return {
      totalReturn,
      sharpeRatio,
      maxDrawdown,
      winRate,
      profitFactor,
      calmarRatio,
      sortinoRatio
    };
  },

  calculateBollingerBands(prices, period, stdMultiplier) {
    const sma = this.calculateSma(prices, period);
    const bands = [];

    for (let i = 0; i < sma.length; i++) {
      const window = prices.slice(i, i + period);
      const stdDev = this.calculateStdDev(window);
      const middle = sma[i];
      const upper = middle + (stdMultiplier * stdDev);
      const lower = middle - (stdMultiplier * stdDev);
      bands.push([lower, middle, upper]);
    }

    return bands;
  },

  calculateRsi(prices, period) {
    if (prices.length < period + 1) return [];

    const gains = [];
    const losses = [];

    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }

    let avgGain = gains.slice(0, period).reduce((sum, g) => sum + g, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((sum, l) => sum + l, 0) / period;

    const rsiValues = [];

    for (let i = period; i < gains.length; i++) {
      avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
      avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;

      const rs = avgLoss !== 0 ? avgGain / avgLoss : 100;
      const rsi = 100 - (100 / (1 + rs));
      rsiValues.push(rsi);
    }

    return rsiValues;
  },

  calculateCorrelation(series1, series2) {
    if (series1.length !== series2.length || series1.length === 0) return 0;

    const n = series1.length;
    const mean1 = series1.reduce((sum, val) => sum + val, 0) / n;
    const mean2 = series2.reduce((sum, val) => sum + val, 0) / n;

    let covariance = 0;
    let var1 = 0;
    let var2 = 0;

    for (let i = 0; i < n; i++) {
      const diff1 = series1[i] - mean1;
      const diff2 = series2[i] - mean2;
      covariance += diff1 * diff2;
      var1 += diff1 * diff1;
      var2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(var1 * var2);
    return denominator === 0 ? 0 : covariance / denominator;
  },

  fastBacktest(bars, signals, initialCapital, commission) {
    if (bars.length !== signals.length) {
      throw new Error('Bars and signals length mismatch');
    }

    const equityCurve = [];
    let position = 0;
    let cash = initialCapital;

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const signal = signals[i];

      if (signal > 0 && position === 0) {
        const shares = (cash * 0.95) / bar.close;
        const cost = shares * bar.close * (1 + commission);
        if (cost <= cash) {
          position = shares;
          cash -= cost;
        }
      } else if (signal < 0 && position > 0) {
        const proceeds = position * bar.close * (1 - commission);
        cash += proceeds;
        position = 0;
      }

      const equity = cash + (position * bar.close);
      equityCurve.push(equity);
    }

    return equityCurve;
  }
};

// Export unified API (native or fallback)
module.exports = {
  calculateSma: (prices, period) =>
    nativeModule ? nativeModule.calculateSma(prices, period) : fallbacks.calculateSma(prices, period),

  calculateEma: (prices, period) =>
    nativeModule ? nativeModule.calculateEma(prices, period) : fallbacks.calculateEma(prices, period),

  calculateReturns: (prices) =>
    nativeModule ? nativeModule.calculateReturns(prices) : fallbacks.calculateReturns(prices),

  calculateStdDev: (values) =>
    nativeModule ? nativeModule.calculateStdDev(values) : fallbacks.calculateStdDev(values),

  calculateSharpeRatio: (returns, riskFreeRate = 0) =>
    nativeModule ? nativeModule.calculateSharpeRatio(returns, riskFreeRate) : fallbacks.calculateSharpeRatio(returns, riskFreeRate),

  calculateMaxDrawdown: (equityCurve) =>
    nativeModule ? nativeModule.calculateMaxDrawdown(equityCurve) : fallbacks.calculateMaxDrawdown(equityCurve),

  calculatePerformanceMetrics: (equityCurve, returns, trades, wins, totalProfit, totalLoss) =>
    nativeModule ? nativeModule.calculatePerformanceMetrics(equityCurve, returns, trades, wins, totalProfit, totalLoss) : fallbacks.calculatePerformanceMetrics(equityCurve, returns, trades, wins, totalProfit, totalLoss),

  calculateBollingerBands: (prices, period, stdMultiplier) =>
    nativeModule ? nativeModule.calculateBollingerBands(prices, period, stdMultiplier) : fallbacks.calculateBollingerBands(prices, period, stdMultiplier),

  calculateRsi: (prices, period) =>
    nativeModule ? nativeModule.calculateRsi(prices, period) : fallbacks.calculateRsi(prices, period),

  calculateCorrelation: (series1, series2) =>
    nativeModule ? nativeModule.calculateCorrelation(series1, series2) : fallbacks.calculateCorrelation(series1, series2),

  fastBacktest: (bars, signals, initialCapital, commission) =>
    nativeModule ? nativeModule.fastBacktest(bars, signals, initialCapital, commission) : fallbacks.fastBacktest(bars, signals, initialCapital, commission),

  isNativeAvailable: () => nativeModule !== null
};
