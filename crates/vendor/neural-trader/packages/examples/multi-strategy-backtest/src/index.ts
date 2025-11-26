/**
 * Multi-Strategy Backtesting System
 * Self-learning strategy allocation with reinforcement learning and swarm optimization
 */

import { OpenAI } from 'openai';
import { Backtester } from './backtester';
import { StrategyLearner } from './strategy-learner';
import { SwarmOptimizer } from './swarm-optimizer';
import { MomentumStrategy } from './strategies/momentum';
import { MeanReversionStrategy } from './strategies/mean-reversion';
import { PairsTradingStrategy } from './strategies/pairs-trading';
import { MarketMakingStrategy } from './strategies/market-making';
import {
  BacktestConfig,
  MarketData,
  StrategyPerformance,
  Strategy
} from './types';

export * from './backtester';
export * from './strategy-learner';
export * from './swarm-optimizer';
export * from './types';
export * from './strategies/momentum';
export * from './strategies/mean-reversion';
export * from './strategies/pairs-trading';
export * from './strategies/market-making';

/**
 * Main orchestrator for multi-strategy backtesting
 */
export class MultiStrategyBacktestSystem {
  private backtester?: Backtester;
  private learner: StrategyLearner;
  private optimizer: SwarmOptimizer;
  private strategies: Strategy[];
  private openai?: OpenAI;

  constructor(
    private config: BacktestConfig,
    openRouterKey?: string
  ) {
    // Initialize strategies
    this.strategies = [
      new MomentumStrategy(),
      new MeanReversionStrategy(),
      new PairsTradingStrategy(),
      new MarketMakingStrategy()
    ];

    // Initialize learner and optimizer
    this.learner = new StrategyLearner({
      learningRate: 0.1,
      discountFactor: 0.95,
      explorationRate: 0.3,
      experienceBufferSize: 5000
    });

    this.optimizer = new SwarmOptimizer({
      particleCount: 20,
      maxIterations: 50,
      inertia: 0.7
    });

    // Initialize OpenRouter for strategy discovery if key provided
    if (openRouterKey) {
      this.openai = new OpenAI({
        baseURL: 'https://openrouter.ai/api/v1',
        apiKey: openRouterKey
      });
    }
  }

  /**
   * Initialize the system
   */
  async initialize(): Promise<void> {
    console.log('🚀 Initializing Multi-Strategy Backtest System...');
    await this.learner.initialize();
    console.log('✅ System initialized');
  }

  /**
   * Run complete backtesting workflow with learning and optimization
   */
  async runCompleteWorkflow(marketData: MarketData[]): Promise<{
    performances: StrategyPerformance[];
    learningStats: any;
    optimizationResults: Map<string, any>;
  }> {
    console.log('\n' + '='.repeat(80));
    console.log('🎯 MULTI-STRATEGY BACKTEST WITH REINFORCEMENT LEARNING');
    console.log('='.repeat(80));

    // Step 1: Initial backtest
    console.log('\n📊 Phase 1: Initial Backtest');
    this.backtester = new Backtester(this.config, this.strategies);
    let performances = await this.backtester.runBacktest(marketData);
    this.displayPerformances(performances);

    // Step 2: Learn from results
    console.log('\n🧠 Phase 2: Reinforcement Learning');
    const portfolioStates = this.backtester.getPortfolioStates();
    await this.learner.learnFromBacktest(portfolioStates, performances);
    const learningStats = this.learner.getStats();
    this.displayLearningStats(learningStats);

    // Step 3: Optimize strategy parameters
    console.log('\n🐝 Phase 3: Swarm Optimization');
    const optimizationResults = await this.optimizer.optimizeMultipleStrategies(
      this.config,
      marketData,
      this.strategies
    );
    this.displayOptimizationResults(optimizationResults);

    // Step 4: Re-run backtest with optimized parameters
    console.log('\n📈 Phase 4: Final Backtest with Optimized Parameters');
    this.backtester = new Backtester(this.config, this.strategies);
    performances = await this.backtester.runBacktest(marketData);
    this.displayPerformances(performances);

    // Step 5: Learn from improved results
    console.log('\n🎓 Phase 5: Learning from Optimized Results');
    await this.learner.learnFromBacktest(
      this.backtester.getPortfolioStates(),
      performances
    );

    // Step 6: Discover new strategies using AI (if OpenRouter configured)
    if (this.openai) {
      console.log('\n🤖 Phase 6: AI Strategy Discovery');
      await this.discoverNewStrategies(performances, marketData);
    }

    console.log('\n' + '='.repeat(80));
    console.log('✅ COMPLETE WORKFLOW FINISHED');
    console.log('='.repeat(80));

    return {
      performances,
      learningStats: this.learner.getStats(),
      optimizationResults
    };
  }

  /**
   * Use OpenRouter to discover new trading strategies
   */
  private async discoverNewStrategies(
    currentPerformances: StrategyPerformance[],
    marketData: MarketData[]
  ): Promise<void> {
    if (!this.openai) return;

    try {
      console.log('🔍 Analyzing market conditions and current strategies...');

      const prompt = this.buildStrategyDiscoveryPrompt(currentPerformances, marketData);

      const completion = await this.openai.chat.completions.create({
        model: 'anthropic/claude-3.5-sonnet',
        messages: [
          {
            role: 'system',
            content: 'You are an expert quantitative trader and strategy researcher. Analyze trading performance and market conditions to suggest innovative strategy improvements.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 2000
      });

      const suggestions = completion.choices[0]?.message?.content;
      if (suggestions) {
        console.log('\n💡 AI Strategy Suggestions:');
        console.log('─'.repeat(80));
        console.log(suggestions);
        console.log('─'.repeat(80));
      }
    } catch (error) {
      console.error('❌ Error in AI strategy discovery:', error);
    }
  }

  /**
   * Build prompt for strategy discovery
   */
  private buildStrategyDiscoveryPrompt(
    performances: StrategyPerformance[],
    marketData: MarketData[]
  ): string {
    const perfSummary = performances.map(p =>
      `${p.strategyName}: Sharpe ${p.sharpeRatio.toFixed(2)}, Return ${(p.totalReturn * 100).toFixed(2)}%, Win Rate ${(p.winRate * 100).toFixed(1)}%`
    ).join('\n');

    const recentPrices = marketData.slice(-100).map(d => d.close);
    const volatility = this.calculateVolatility(recentPrices);
    const trend = this.calculateTrend(recentPrices);

    return `
Analyze the following trading strategy performances and market conditions:

CURRENT STRATEGY PERFORMANCES:
${perfSummary}

MARKET CONDITIONS:
- Recent volatility: ${(volatility * 100).toFixed(2)}%
- Trend: ${trend > 0 ? 'Bullish' : 'Bearish'} (${(Math.abs(trend) * 100).toFixed(2)}%)
- Data points: ${marketData.length}

Please provide:
1. Analysis of which strategies are working well and why
2. Suggestions for parameter adjustments to underperforming strategies
3. Ideas for new strategy types that might work in current market conditions
4. Risk management improvements
5. Specific actionable recommendations for the next optimization cycle

Focus on practical, implementable suggestions that can improve Sharpe ratio and reduce drawdowns.
    `.trim();
  }

  /**
   * Display strategy performances
   */
  private displayPerformances(performances: StrategyPerformance[]): void {
    console.log('\n📊 Strategy Performance Summary:');
    console.log('─'.repeat(100));
    console.log(
      'Strategy'.padEnd(20) +
      'Return'.padEnd(12) +
      'Sharpe'.padEnd(10) +
      'MaxDD'.padEnd(10) +
      'Win Rate'.padEnd(12) +
      'Trades'.padEnd(10) +
      'Profit Factor'
    );
    console.log('─'.repeat(100));

    for (const perf of performances) {
      console.log(
        perf.strategyName.padEnd(20) +
        `${(perf.totalReturn * 100).toFixed(2)}%`.padEnd(12) +
        perf.sharpeRatio.toFixed(2).padEnd(10) +
        `${(perf.maxDrawdown * 100).toFixed(2)}%`.padEnd(10) +
        `${(perf.winRate * 100).toFixed(1)}%`.padEnd(12) +
        perf.trades.toString().padEnd(10) +
        perf.profitFactor.toFixed(2)
      );
    }
    console.log('─'.repeat(100));
  }

  /**
   * Display learning statistics
   */
  private displayLearningStats(stats: any): void {
    console.log('\n🧠 Learning Statistics:');
    console.log('─'.repeat(60));
    console.log(`Episodes: ${stats.episodes}`);
    console.log(`Total Reward: ${stats.totalReward.toFixed(2)}`);
    console.log(`Exploration Rate: ${(stats.explorationRate * 100).toFixed(2)}%`);
    console.log(`Q-Table Size: ${stats.qTableSize} states`);
    console.log(`Experience Buffer: ${stats.experienceCount} experiences`);
    console.log('─'.repeat(60));

    if (stats.bestPerformances.length > 0) {
      console.log('\n🏆 Best Historical Performances:');
      for (const perf of stats.bestPerformances) {
        console.log(`  ${perf.strategyName}: Sharpe ${perf.sharpeRatio.toFixed(2)}`);
      }
    }
  }

  /**
   * Display optimization results
   */
  private displayOptimizationResults(results: Map<string, any>): void {
    console.log('\n🎯 Optimization Results:');
    console.log('─'.repeat(80));

    for (const [strategy, result] of results.entries()) {
      console.log(`\n${strategy}:`);
      console.log(`  Best Score: ${result.bestScore.toFixed(4)}`);
      console.log(`  Evaluations: ${result.evaluations}`);
      console.log(`  Time: ${(result.timeElapsed / 1000).toFixed(2)}s`);
      console.log(`  Parameters:`, result.bestParameters);
    }
    console.log('─'.repeat(80));
  }

  // Utility methods
  private calculateVolatility(prices: number[]): number {
    const returns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const squaredDiffs = returns.map(r => Math.pow(r - mean, 2));
    const variance = squaredDiffs.reduce((sum, d) => sum + d, 0) / returns.length;

    return Math.sqrt(variance) * Math.sqrt(252); // Annualized
  }

  private calculateTrend(prices: number[]): number {
    if (prices.length < 2) return 0;
    return (prices[prices.length - 1] - prices[0]) / prices[0];
  }

  /**
   * Get current strategies
   */
  getStrategies(): Strategy[] {
    return this.strategies;
  }

  /**
   * Get learner instance
   */
  getLearner(): StrategyLearner {
    return this.learner;
  }

  /**
   * Get optimizer instance
   */
  getOptimizer(): SwarmOptimizer {
    return this.optimizer;
  }
}

// Example usage
if (require.main === module) {
  (async () => {
    // Generate sample market data
    const marketData: MarketData[] = [];
    let price = 100;
    const startTime = Date.now() - (365 * 24 * 60 * 60 * 1000);

    for (let i = 0; i < 1000; i++) {
      const change = (Math.random() - 0.48) * 2;
      price *= (1 + change / 100);

      marketData.push({
        timestamp: startTime + (i * 24 * 60 * 60 * 1000),
        symbol: 'TEST',
        open: price * 0.99,
        high: price * 1.02,
        low: price * 0.98,
        close: price,
        volume: Math.floor(Math.random() * 1000000) + 500000
      });
    }

    // Configure backtest
    const config: BacktestConfig = {
      startDate: new Date(marketData[0].timestamp),
      endDate: new Date(marketData[marketData.length - 1].timestamp),
      initialCapital: 100000,
      symbols: ['TEST'],
      strategies: [
        { name: 'momentum', type: 'momentum', initialWeight: 0.25, parameters: {}, enabled: true },
        { name: 'mean-reversion', type: 'mean-reversion', initialWeight: 0.25, parameters: {}, enabled: true },
        { name: 'pairs-trading', type: 'pairs-trading', initialWeight: 0.25, parameters: {}, enabled: true },
        { name: 'market-making', type: 'market-making', initialWeight: 0.25, parameters: {}, enabled: true }
      ],
      commission: 0.001,
      slippage: 0.0005,
      walkForwardPeriods: 3,
      rebalanceFrequency: 'weekly'
    };

    // Run system
    const system = new MultiStrategyBacktestSystem(
      config,
      process.env.OPENROUTER_API_KEY
    );

    await system.initialize();
    const results = await system.runCompleteWorkflow(marketData);

    console.log('\n🎉 Backtest Complete!');
    console.log(`📈 Final Performance: ${results.performances.length} strategies evaluated`);
  })();
}
