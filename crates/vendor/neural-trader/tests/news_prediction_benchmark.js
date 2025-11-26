#!/usr/bin/env node
/**
 * Comprehensive News & Prediction Market Tools Benchmark
 * Tests functionality, performance, and accuracy of 10 MCP tools
 */

const { performance } = require('perf_hooks');

// Benchmark configuration
const BENCHMARK_CONFIG = {
  newsArticleCount: 1000,
  sentimentSamples: 100,
  predictionMarkets: 50,
  orderBookDepth: 20,
  iterations: 10,
  warmupRuns: 3,
};

// Test data generator
class TestDataGenerator {
  static generateNewsArticles(count) {
    const headlines = [
      { text: "Company beats earnings expectations with strong Q4 results", sentiment: 0.85 },
      { text: "Stock crashes amid regulatory investigation and losses", sentiment: -0.75 },
      { text: "Market neutral as investors await economic data", sentiment: 0.05 },
      { text: "Bullish rally continues with record profits and growth surge", sentiment: 0.92 },
      { text: "Bearish trend accelerates with declining revenues and layoffs", sentiment: -0.88 },
      { text: "Company announces strategic partnership and expansion plans", sentiment: 0.68 },
      { text: "Downgrade concerns weigh on stock as analysts cut targets", sentiment: -0.62 },
      { text: "Innovative product launch drives investor optimism", sentiment: 0.78 },
      { text: "Bankruptcy fears mount as debt levels increase", sentiment: -0.82 },
      { text: "Trading volume remains stable in quiet session", sentiment: 0.02 },
    ];

    return Array(count).fill(null).map((_, i) => {
      const template = headlines[i % headlines.length];
      return {
        id: `news_${i}`,
        headline: template.text,
        content: `${template.text}. Additional context and details about the market situation and company performance.`,
        symbol: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'][i % 5],
        source: ['Bloomberg', 'Reuters', 'WSJ', 'CNBC', 'FT'][i % 5],
        timestamp: new Date(Date.now() - i * 3600000).toISOString(),
        expectedSentiment: template.sentiment,
      };
    });
  }

  static generatePredictionMarkets(count) {
    return Array(count).fill(null).map((_, i) => ({
      market_id: `pm_${i}`,
      question: `Will event ${i} occur by end of year?`,
      category: ['crypto', 'politics', 'sports', 'tech', 'economy'][i % 5],
      current_price: 0.45 + (i % 30) * 0.01,
      volume: 100000 + i * 10000,
      liquidity: 50000 + i * 5000,
    }));
  }
}

// Performance metrics collector
class PerformanceMetrics {
  constructor() {
    this.metrics = new Map();
  }

  record(operation, duration, metadata = {}) {
    if (!this.metrics.has(operation)) {
      this.metrics.set(operation, []);
    }
    this.metrics.get(operation).push({ duration, ...metadata });
  }

  getStats(operation) {
    const data = this.metrics.get(operation) || [];
    if (data.length === 0) return null;

    const durations = data.map(d => d.duration);
    durations.sort((a, b) => a - b);

    return {
      count: data.length,
      min: durations[0],
      max: durations[durations.length - 1],
      mean: durations.reduce((a, b) => a + b, 0) / durations.length,
      median: durations[Math.floor(durations.length / 2)],
      p95: durations[Math.floor(durations.length * 0.95)],
      p99: durations[Math.floor(durations.length * 0.99)],
    };
  }

  summary() {
    const summary = {};
    for (const [operation, _] of this.metrics) {
      summary[operation] = this.getStats(operation);
    }
    return summary;
  }
}

// News sentiment accuracy validator
class SentimentAccuracyValidator {
  constructor() {
    this.predictions = [];
    this.groundTruth = [];
  }

  addPrediction(predicted, actual) {
    this.predictions.push(predicted);
    this.groundTruth.push(actual);
  }

  calculateMetrics() {
    if (this.predictions.length === 0) {
      return { mae: 0, rmse: 0, accuracy: 0, correlation: 0 };
    }

    // Mean Absolute Error
    const mae = this.predictions.reduce((sum, pred, i) => {
      return sum + Math.abs(pred - this.groundTruth[i]);
    }, 0) / this.predictions.length;

    // Root Mean Square Error
    const mse = this.predictions.reduce((sum, pred, i) => {
      const diff = pred - this.groundTruth[i];
      return sum + diff * diff;
    }, 0) / this.predictions.length;
    const rmse = Math.sqrt(mse);

    // Classification accuracy (positive/negative/neutral)
    const classify = (score) => {
      if (score > 0.1) return 'positive';
      if (score < -0.1) return 'negative';
      return 'neutral';
    };
    const correctCount = this.predictions.reduce((count, pred, i) => {
      return count + (classify(pred) === classify(this.groundTruth[i]) ? 1 : 0);
    }, 0);
    const accuracy = correctCount / this.predictions.length;

    // Pearson correlation
    const meanPred = this.predictions.reduce((a, b) => a + b, 0) / this.predictions.length;
    const meanTruth = this.groundTruth.reduce((a, b) => a + b, 0) / this.groundTruth.length;

    let numerator = 0, denomPred = 0, denomTruth = 0;
    for (let i = 0; i < this.predictions.length; i++) {
      const predDiff = this.predictions[i] - meanPred;
      const truthDiff = this.groundTruth[i] - meanTruth;
      numerator += predDiff * truthDiff;
      denomPred += predDiff * predDiff;
      denomTruth += truthDiff * truthDiff;
    }
    const correlation = numerator / Math.sqrt(denomPred * denomTruth);

    return { mae, rmse, accuracy, correlation };
  }
}

// Expected value calculation validator
class EVCalculationValidator {
  static validateKellyCalculation(investment, winProb, winMultiplier, loseMultiplier = -1) {
    // Kelly Criterion: f* = (bp - q) / b
    // where b = odds, p = win probability, q = lose probability
    const q = 1 - winProb;
    const b = winMultiplier / Math.abs(loseMultiplier);
    const kellyFraction = (b * winProb - q) / b;

    return {
      kellyFraction,
      recommendedBet: investment * Math.max(0, Math.min(kellyFraction, 0.25)), // Cap at 25%
      expectedValue: (winProb * winMultiplier + q * loseMultiplier) * investment,
    };
  }

  static validateExpectedValue(outcomes) {
    // EV = Σ (probability × payout)
    return outcomes.reduce((ev, outcome) => {
      return ev + (outcome.probability * outcome.payout);
    }, 0);
  }
}

// Mock MCP tool implementations (since we can't call actual MCP in test)
class MockMCPTools {
  static async analyze_news(params) {
    const start = performance.now();

    // Simulate sentiment analysis with lexicon-based approach
    const text = params.headline || params.content || '';
    const words = text.toLowerCase().split(/\s+/);

    const positiveWords = ['beats', 'strong', 'growth', 'bullish', 'rally', 'profit', 'surge', 'record'];
    const negativeWords = ['crash', 'loss', 'decline', 'bearish', 'investigation', 'bankruptcy', 'layoffs'];

    let score = 0;
    words.forEach(word => {
      if (positiveWords.some(pw => word.includes(pw))) score += 0.15;
      if (negativeWords.some(nw => word.includes(nw))) score -= 0.15;
    });

    score = Math.max(-1, Math.min(1, score));

    const duration = performance.now() - start;

    return {
      sentiment: {
        overall_score: score,
        confidence: 0.85,
        trend: score > 0.1 ? 'positive' : score < -0.1 ? 'negative' : 'neutral',
      },
      computation_time_ms: duration,
    };
  }

  static async get_news_sentiment(params) {
    const start = performance.now();

    const result = {
      symbol: params.symbol,
      current_sentiment: 0.72,
      sentiment_trend: 'improving',
      sentiment_distribution: {
        very_bullish: 0.25,
        bullish: 0.35,
        neutral: 0.25,
        bearish: 0.10,
        very_bearish: 0.05,
      },
    };

    const duration = performance.now() - start;
    return { ...result, computation_time_ms: duration };
  }

  static async control_news_collection(params) {
    const start = performance.now();
    // Simulate collection control
    await new Promise(resolve => setTimeout(resolve, 10));
    return { status: 'success', action: params.action, duration: performance.now() - start };
  }

  static async get_news_trends(params) {
    const start = performance.now();
    const trends = params.symbols.map(symbol => ({
      symbol,
      trend_score: Math.random() * 2 - 1,
      momentum: Math.random() * 0.5,
      volume_change: Math.random() * 100,
    }));
    return { trends, duration: performance.now() - start };
  }

  static async get_prediction_markets(params) {
    const start = performance.now();
    const markets = TestDataGenerator.generatePredictionMarkets(params.limit || 10);
    return { markets, duration: performance.now() - start };
  }

  static async analyze_market_sentiment(params) {
    const start = performance.now();
    const result = {
      market_id: params.market_id,
      sentiment_analysis: {
        market_confidence: 0.78,
        trend: 'bullish',
        momentum: 0.12,
      },
    };
    return { ...result, duration: performance.now() - start };
  }

  static async get_market_orderbook(params) {
    const start = performance.now();
    const depth = params.depth || 10;
    const orderbook = {
      bids: Array(depth).fill(null).map((_, i) => ({
        price: 0.65 - i * 0.01,
        size: 1000 + i * 100,
      })),
      asks: Array(depth).fill(null).map((_, i) => ({
        price: 0.66 + i * 0.01,
        size: 950 + i * 80,
      })),
      spread: 0.01,
    };
    return { orderbook, duration: performance.now() - start };
  }

  static async place_prediction_order(params) {
    const start = performance.now();
    const result = {
      order_id: `ord_${Date.now()}`,
      status: 'submitted',
      market_id: params.market_id,
      quantity: params.quantity,
    };
    return { ...result, duration: performance.now() - start };
  }

  static async get_prediction_positions() {
    const start = performance.now();
    const positions = [
      {
        market_id: 'pm_001',
        shares: 500,
        avg_price: 0.62,
        current_price: 0.65,
        unrealized_pnl: 15.0,
      },
    ];
    return { positions, duration: performance.now() - start };
  }

  static async calculate_expected_value(params) {
    const start = performance.now();
    const investment = params.investment_amount;
    const winProb = 0.65;
    const winMultiplier = 1.54;

    const validation = EVCalculationValidator.validateKellyCalculation(
      investment,
      winProb,
      winMultiplier
    );

    return {
      expected_value: validation.expectedValue,
      kelly_recommendation: validation.recommendedBet,
      duration: performance.now() - start,
    };
  }
}

// Main benchmark suite
class NewsPredictionBenchmark {
  constructor() {
    this.metrics = new PerformanceMetrics();
    this.sentimentValidator = new SentimentAccuracyValidator();
    this.results = {};
  }

  async runAll() {
    console.log('🚀 Starting News & Prediction Market Tools Benchmark\n');
    console.log('Configuration:', BENCHMARK_CONFIG, '\n');

    // Warmup
    console.log('⏱️  Running warmup iterations...');
    for (let i = 0; i < BENCHMARK_CONFIG.warmupRuns; i++) {
      await this.benchmarkNewsSentiment(10);
    }

    // News tools benchmarks
    await this.benchmarkNewsSentiment(BENCHMARK_CONFIG.sentimentSamples);
    await this.benchmarkNewsCollection();
    await this.benchmarkNewsTrends();
    await this.benchmarkSentimentAccuracy();

    // Prediction market benchmarks
    await this.benchmarkPredictionMarkets();
    await this.benchmarkMarketSentiment();
    await this.benchmarkOrderbook();
    await this.benchmarkOrderPlacement();
    await this.benchmarkPositionTracking();
    await this.benchmarkEVCalculation();

    // Load tests
    await this.loadTestNewsSentiment(BENCHMARK_CONFIG.newsArticleCount);

    this.generateReport();
  }

  async benchmarkNewsSentiment(count) {
    console.log(`\n📰 Benchmarking news sentiment analysis (${count} articles)...`);

    const articles = TestDataGenerator.generateNewsArticles(count);

    for (const article of articles) {
      const start = performance.now();
      const result = await MockMCPTools.analyze_news(article);
      const duration = performance.now() - start;

      this.metrics.record('analyze_news', duration, {
        symbol: article.symbol,
        sentimentScore: result.sentiment.overall_score,
      });

      this.sentimentValidator.addPrediction(
        result.sentiment.overall_score,
        article.expectedSentiment
      );
    }

    const stats = this.metrics.getStats('analyze_news');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms | P99: ${stats.p99.toFixed(2)}ms`);
  }

  async benchmarkNewsCollection() {
    console.log('\n📡 Benchmarking news collection control...');

    const actions = ['start', 'stop', 'pause', 'resume'];

    for (const action of actions) {
      for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
        const start = performance.now();
        await MockMCPTools.control_news_collection({
          action,
          symbols: ['AAPL', 'GOOGL', 'MSFT'],
        });
        const duration = performance.now() - start;
        this.metrics.record('control_news_collection', duration, { action });
      }
    }

    const stats = this.metrics.getStats('control_news_collection');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkNewsTrends() {
    console.log('\n📈 Benchmarking news trend analysis...');

    for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
      const start = performance.now();
      await MockMCPTools.get_news_trends({
        symbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
      });
      const duration = performance.now() - start;
      this.metrics.record('get_news_trends', duration);
    }

    const stats = this.metrics.getStats('get_news_trends');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkSentimentAccuracy() {
    console.log('\n🎯 Validating sentiment accuracy...');

    const accuracy = this.sentimentValidator.calculateMetrics();

    console.log(`  ✓ MAE: ${accuracy.mae.toFixed(4)}`);
    console.log(`  ✓ RMSE: ${accuracy.rmse.toFixed(4)}`);
    console.log(`  ✓ Classification Accuracy: ${(accuracy.accuracy * 100).toFixed(2)}%`);
    console.log(`  ✓ Correlation: ${accuracy.correlation.toFixed(4)}`);

    this.results.sentimentAccuracy = accuracy;
  }

  async benchmarkPredictionMarkets() {
    console.log('\n🎲 Benchmarking prediction market listing...');

    for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
      const start = performance.now();
      await MockMCPTools.get_prediction_markets({ limit: 50 });
      const duration = performance.now() - start;
      this.metrics.record('get_prediction_markets', duration);
    }

    const stats = this.metrics.getStats('get_prediction_markets');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkMarketSentiment() {
    console.log('\n💹 Benchmarking market sentiment analysis...');

    for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
      const start = performance.now();
      await MockMCPTools.analyze_market_sentiment({
        market_id: `pm_${i}`,
        use_gpu: false,
      });
      const duration = performance.now() - start;
      this.metrics.record('analyze_market_sentiment', duration);
    }

    const stats = this.metrics.getStats('analyze_market_sentiment');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkOrderbook() {
    console.log('\n📊 Benchmarking orderbook retrieval...');

    for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
      const start = performance.now();
      await MockMCPTools.get_market_orderbook({
        market_id: 'pm_001',
        depth: BENCHMARK_CONFIG.orderBookDepth,
      });
      const duration = performance.now() - start;
      this.metrics.record('get_market_orderbook', duration);
    }

    const stats = this.metrics.getStats('get_market_orderbook');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkOrderPlacement() {
    console.log('\n💰 Benchmarking order placement...');

    for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
      const start = performance.now();
      await MockMCPTools.place_prediction_order({
        market_id: 'pm_001',
        outcome: 'Yes',
        side: 'buy',
        quantity: 100,
      });
      const duration = performance.now() - start;
      this.metrics.record('place_prediction_order', duration);
    }

    const stats = this.metrics.getStats('place_prediction_order');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkPositionTracking() {
    console.log('\n📍 Benchmarking position tracking...');

    for (let i = 0; i < BENCHMARK_CONFIG.iterations; i++) {
      const start = performance.now();
      await MockMCPTools.get_prediction_positions();
      const duration = performance.now() - start;
      this.metrics.record('get_prediction_positions', duration);
    }

    const stats = this.metrics.getStats('get_prediction_positions');
    console.log(`  ✓ Mean: ${stats.mean.toFixed(2)}ms | P95: ${stats.p95.toFixed(2)}ms`);
  }

  async benchmarkEVCalculation() {
    console.log('\n🧮 Validating EV calculations...');

    const testCases = [
      { investment: 100, winProb: 0.65, winMult: 1.54, expected: 12 },
      { investment: 1000, winProb: 0.52, winMult: 1.92, expected: 0 },
      { investment: 500, winProb: 0.75, winMult: 1.33, expected: 0 },
    ];

    let totalError = 0;
    let validationCount = 0;

    for (const testCase of testCases) {
      const start = performance.now();
      const result = await MockMCPTools.calculate_expected_value({
        investment_amount: testCase.investment,
      });
      const duration = performance.now() - start;

      this.metrics.record('calculate_expected_value', duration);

      // Validate Kelly calculation
      const validation = EVCalculationValidator.validateKellyCalculation(
        testCase.investment,
        testCase.winProb,
        testCase.winMult
      );

      const error = Math.abs(validation.expectedValue - result.expected_value);
      totalError += error;
      validationCount++;

      console.log(`  Case ${validationCount}:`);
      console.log(`    Expected EV: $${validation.expectedValue.toFixed(2)}`);
      console.log(`    Kelly Bet: $${validation.recommendedBet.toFixed(2)}`);
      console.log(`    Error: $${error.toFixed(2)}`);
    }

    const stats = this.metrics.getStats('calculate_expected_value');
    console.log(`  ✓ Mean latency: ${stats.mean.toFixed(2)}ms`);
    console.log(`  ✓ Average error: $${(totalError / validationCount).toFixed(2)}`);

    this.results.evValidation = {
      averageError: totalError / validationCount,
      testCases: validationCount,
    };
  }

  async loadTestNewsSentiment(articleCount) {
    console.log(`\n🔥 Load test: ${articleCount} news articles...`);

    const articles = TestDataGenerator.generateNewsArticles(articleCount);
    const batchSize = 100;
    const batches = Math.ceil(articles.length / batchSize);

    let totalProcessed = 0;
    const overallStart = performance.now();

    for (let b = 0; b < batches; b++) {
      const batch = articles.slice(b * batchSize, (b + 1) * batchSize);
      const batchStart = performance.now();

      await Promise.all(
        batch.map(article => MockMCPTools.analyze_news(article))
      );

      const batchDuration = performance.now() - batchStart;
      totalProcessed += batch.length;

      this.metrics.record('batch_sentiment_analysis', batchDuration, {
        batchSize: batch.length,
      });
    }

    const totalDuration = performance.now() - overallStart;
    const throughput = (totalProcessed / totalDuration) * 1000; // articles per second

    console.log(`  ✓ Processed ${totalProcessed} articles in ${totalDuration.toFixed(2)}ms`);
    console.log(`  ✓ Throughput: ${throughput.toFixed(2)} articles/sec`);

    this.results.loadTest = {
      articlesProcessed: totalProcessed,
      totalDuration,
      throughput,
    };
  }

  generateReport() {
    console.log('\n' + '='.repeat(80));
    console.log('📊 BENCHMARK SUMMARY REPORT');
    console.log('='.repeat(80) + '\n');

    const summary = this.metrics.summary();

    console.log('Performance Metrics (in milliseconds):');
    console.log('-'.repeat(80));
    console.log('Operation'.padEnd(35), 'Mean', 'P95', 'P99', 'Count');
    console.log('-'.repeat(80));

    for (const [operation, stats] of Object.entries(summary)) {
      if (stats) {
        console.log(
          operation.padEnd(35),
          stats.mean.toFixed(2).padStart(8),
          stats.p95.toFixed(2).padStart(8),
          stats.p99.toFixed(2).padStart(8),
          stats.count.toString().padStart(8)
        );
      }
    }

    console.log('\n' + '='.repeat(80));
    console.log('Accuracy Validation:');
    console.log('-'.repeat(80));

    if (this.results.sentimentAccuracy) {
      const acc = this.results.sentimentAccuracy;
      console.log(`Sentiment Analysis MAE:            ${acc.mae.toFixed(4)}`);
      console.log(`Sentiment Analysis RMSE:           ${acc.rmse.toFixed(4)}`);
      console.log(`Classification Accuracy:           ${(acc.accuracy * 100).toFixed(2)}%`);
      console.log(`Pearson Correlation:               ${acc.correlation.toFixed(4)}`);
    }

    if (this.results.evValidation) {
      const ev = this.results.evValidation;
      console.log(`EV Calculation Average Error:      $${ev.averageError.toFixed(2)}`);
      console.log(`EV Test Cases Validated:           ${ev.testCases}`);
    }

    console.log('\n' + '='.repeat(80));
    console.log('Load Test Results:');
    console.log('-'.repeat(80));

    if (this.results.loadTest) {
      const lt = this.results.loadTest;
      console.log(`Articles Processed:                ${lt.articlesProcessed}`);
      console.log(`Total Duration:                    ${lt.totalDuration.toFixed(2)}ms`);
      console.log(`Throughput:                        ${lt.throughput.toFixed(2)} articles/sec`);
    }

    console.log('\n' + '='.repeat(80));
    console.log('✅ Benchmark Complete');
    console.log('='.repeat(80) + '\n');

    // Export results to JSON
    this.exportResults();
  }

  exportResults() {
    const exportData = {
      timestamp: new Date().toISOString(),
      config: BENCHMARK_CONFIG,
      metrics: this.metrics.summary(),
      validation: {
        sentiment: this.results.sentimentAccuracy,
        expectedValue: this.results.evValidation,
      },
      loadTest: this.results.loadTest,
    };

    const fs = require('fs');
    const path = '/workspaces/neural-trader/docs/mcp-analysis/benchmark_results.json';
    fs.writeFileSync(path, JSON.stringify(exportData, null, 2));
    console.log(`📁 Results exported to: ${path}\n`);
  }
}

// Run the benchmark
if (require.main === module) {
  const benchmark = new NewsPredictionBenchmark();
  benchmark.runAll().catch(console.error);
}

module.exports = { NewsPredictionBenchmark, TestDataGenerator, EVCalculationValidator };
