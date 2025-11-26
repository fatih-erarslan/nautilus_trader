/**
 * Benchmark Swarm for Portfolio Optimization
 * Explores different algorithms, constraints, and parameters concurrently
 * Uses OpenRouter via agentic-flow for strategy suggestions
 */

import { OpenAI } from 'openai';
import {
  MeanVarianceOptimizer,
  RiskParityOptimizer,
  BlackLittermanOptimizer,
  MultiObjectiveOptimizer,
  OptimizationResult,
  PortfolioConstraints,
  Asset,
} from './optimizer.js';
import { SelfLearningOptimizer } from './self-learning.js';

export interface BenchmarkConfig {
  algorithms: string[];
  constraintVariations: PortfolioConstraints[];
  assets: Asset[];
  correlationMatrix: number[][];
  historicalReturns?: number[][];
  marketCapWeights?: number[];
  iterations?: number;
}

export interface BenchmarkResult {
  algorithm: string;
  constraints: PortfolioConstraints;
  result: OptimizationResult;
  executionTime: number;
  convergenceMetrics?: {
    iterations: number;
    finalGradientNorm: number;
  };
}

export interface SwarmInsights {
  bestAlgorithm: string;
  bestResult: BenchmarkResult;
  algorithmRankings: Array<{ algorithm: string, avgSharpe: number, avgRisk: number }>;
  constraintImpact: Record<string, number>;
  recommendations: string[];
}

/**
 * Portfolio Optimization Swarm
 * Runs multiple optimization strategies concurrently
 */
export class PortfolioOptimizationSwarm {
  private openai: OpenAI | null = null;
  private learningOptimizer?: SelfLearningOptimizer;

  constructor(
    openRouterApiKey?: string,
    learningOptimizer?: SelfLearningOptimizer,
  ) {
    if (openRouterApiKey) {
      this.openai = new OpenAI({
        baseURL: 'https://openrouter.ai/api/v1',
        apiKey: openRouterApiKey,
        defaultHeaders: {
          'HTTP-Referer': 'https://neural-trader.ai',
          'X-Title': 'Neural Trader Portfolio Optimization',
        },
      });
    }
    this.learningOptimizer = learningOptimizer;
  }

  /**
   * Run comprehensive benchmark across all algorithms and constraints
   */
  async runBenchmark(config: BenchmarkConfig): Promise<SwarmInsights> {
    console.log('🚀 Starting Portfolio Optimization Swarm...');
    console.log(`📊 Testing ${config.algorithms.length} algorithms with ${config.constraintVariations.length} constraint sets`);

    const results: BenchmarkResult[] = [];
    const startTime = Date.now();

    // Run all combinations concurrently
    const tasks = [];
    for (const algorithm of config.algorithms) {
      for (const constraints of config.constraintVariations) {
        tasks.push(
          this.runOptimization(algorithm, config, constraints)
        );
      }
    }

    const completedResults = await Promise.allSettled(tasks);

    for (const result of completedResults) {
      if (result.status === 'fulfilled') {
        results.push(result.value);
      } else {
        console.error('❌ Optimization failed:', result.reason);
      }
    }

    const totalTime = Date.now() - startTime;
    console.log(`✅ Completed ${results.length} optimizations in ${totalTime}ms`);

    // Analyze results
    const insights = this.analyzeResults(results, config);

    // Get AI recommendations if OpenRouter is available
    if (this.openai) {
      insights.recommendations = await this.getAIRecommendations(insights, config);
    }

    // Learn from results if learning optimizer is available
    if (this.learningOptimizer) {
      await this.learnFromBenchmark(results);
    }

    return insights;
  }

  /**
   * Run single optimization with timing
   */
  private async runOptimization(
    algorithm: string,
    config: BenchmarkConfig,
    constraints: PortfolioConstraints,
  ): Promise<BenchmarkResult> {
    const startTime = Date.now();
    let result: OptimizationResult;

    switch (algorithm) {
      case 'mean-variance': {
        const optimizer = new MeanVarianceOptimizer(config.assets, config.correlationMatrix);
        result = optimizer.optimize(constraints);
        break;
      }
      case 'risk-parity': {
        const optimizer = new RiskParityOptimizer(config.assets, config.correlationMatrix);
        result = optimizer.optimize(constraints);
        break;
      }
      case 'black-litterman': {
        const marketCapWeights = config.marketCapWeights || Array(config.assets.length).fill(1 / config.assets.length);
        const optimizer = new BlackLittermanOptimizer(
          config.assets,
          config.correlationMatrix,
          marketCapWeights,
        );
        result = optimizer.optimize([], constraints);
        break;
      }
      case 'multi-objective': {
        const historicalReturns = config.historicalReturns || this.generateMockReturns(config.assets.length);
        const optimizer = new MultiObjectiveOptimizer(
          config.assets,
          config.correlationMatrix,
          historicalReturns,
        );
        result = optimizer.optimize({ return: 1.0, risk: 1.0, drawdown: 0.5 }, constraints);
        break;
      }
      default:
        throw new Error(`Unknown algorithm: ${algorithm}`);
    }

    const executionTime = Date.now() - startTime;

    return {
      algorithm,
      constraints,
      result,
      executionTime,
    };
  }

  /**
   * Generate mock historical returns for testing
   */
  private generateMockReturns(numAssets: number, numPeriods: number = 252): number[][] {
    const returns: number[][] = [];
    for (let t = 0; t < numPeriods; t++) {
      const periodReturns = Array(numAssets).fill(0).map(() =>
        (Math.random() - 0.5) * 0.02 // -1% to +1% daily returns
      );
      returns.push(periodReturns);
    }
    return returns;
  }

  /**
   * Analyze benchmark results and generate insights
   */
  private analyzeResults(results: BenchmarkResult[], config: BenchmarkConfig): SwarmInsights {
    // Find best result by Sharpe ratio
    const bestResult = results.reduce((best, current) =>
      current.result.sharpeRatio > best.result.sharpeRatio ? current : best
    );

    // Calculate algorithm rankings
    const algorithmStats = new Map<string, { sharpes: number[], risks: number[] }>();

    for (const result of results) {
      if (!algorithmStats.has(result.algorithm)) {
        algorithmStats.set(result.algorithm, { sharpes: [], risks: [] });
      }
      const stats = algorithmStats.get(result.algorithm)!;
      stats.sharpes.push(result.result.sharpeRatio);
      stats.risks.push(result.result.risk);
    }

    const algorithmRankings = Array.from(algorithmStats.entries())
      .map(([algorithm, stats]) => ({
        algorithm,
        avgSharpe: stats.sharpes.reduce((a, b) => a + b, 0) / stats.sharpes.length,
        avgRisk: stats.risks.reduce((a, b) => a + b, 0) / stats.risks.length,
      }))
      .sort((a, b) => b.avgSharpe - a.avgSharpe);

    // Analyze constraint impact
    const constraintImpact: Record<string, number> = {};

    // Compare results with different constraints
    for (let i = 0; i < config.constraintVariations.length; i++) {
      const constraintResults = results.filter((_, idx) => idx % config.constraintVariations.length === i);
      const avgSharpe = constraintResults.reduce((sum, r) => sum + r.result.sharpeRatio, 0) / constraintResults.length;
      constraintImpact[`constraint_set_${i}`] = avgSharpe;
    }

    return {
      bestAlgorithm: bestResult.algorithm,
      bestResult,
      algorithmRankings,
      constraintImpact,
      recommendations: [],
    };
  }

  /**
   * Get AI-powered strategy recommendations via OpenRouter
   */
  private async getAIRecommendations(
    insights: SwarmInsights,
    config: BenchmarkConfig,
  ): Promise<string[]> {
    if (!this.openai) return [];

    const prompt = `
You are a portfolio optimization expert. Analyze these benchmark results and provide strategic recommendations:

Best Algorithm: ${insights.bestAlgorithm}
Best Sharpe Ratio: ${insights.bestResult.result.sharpeRatio.toFixed(4)}
Best Risk: ${insights.bestResult.result.risk.toFixed(4)}

Algorithm Rankings:
${insights.algorithmRankings.map(r => `- ${r.algorithm}: Avg Sharpe ${r.avgSharpe.toFixed(4)}, Avg Risk ${r.avgRisk.toFixed(4)}`).join('\n')}

Assets:
${config.assets.map(a => `- ${a.symbol}: Expected Return ${a.expectedReturn.toFixed(4)}, Volatility ${a.volatility.toFixed(4)}`).join('\n')}

Provide 3-5 specific, actionable recommendations for improving portfolio performance. Focus on:
1. Algorithm selection based on market conditions
2. Constraint adjustments
3. Risk management strategies
4. Diversification improvements
5. Rebalancing frequency

Format as a JSON array of strings.
`;

    try {
      const response = await this.openai.chat.completions.create({
        model: 'anthropic/claude-3.5-sonnet',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
        max_tokens: 1000,
      });

      const content = response.choices[0]?.message?.content;
      if (!content) return [];

      // Try to parse as JSON array
      const jsonMatch = content.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }

      // Fallback: split by newlines and filter
      return content.split('\n').filter(line => line.trim().length > 0);
    } catch (error) {
      console.error('❌ Failed to get AI recommendations:', error);
      return [];
    }
  }

  /**
   * Learn from benchmark results using self-learning optimizer
   */
  private async learnFromBenchmark(results: BenchmarkResult[]): Promise<void> {
    if (!this.learningOptimizer) return;

    console.log('🧠 Learning from benchmark results...');

    for (const result of results) {
      // Convert result to performance metrics
      const performance = {
        sharpeRatio: result.result.sharpeRatio,
        maxDrawdown: 0.15, // Estimated
        volatility: result.result.risk,
        cumulativeReturn: result.result.expectedReturn,
        winRate: 0.6, // Estimated
        informationRatio: result.result.sharpeRatio * 0.8,
      };

      const marketConditions = {
        volatility: result.result.risk,
        trend: result.result.expectedReturn > 0 ? 1 : -1,
        correlation: 0.5,
      };

      await this.learningOptimizer.learn(result.result, performance, marketConditions);
    }

    console.log('✅ Learning complete');
  }

  /**
   * Explore constraint combinations using swarm intelligence
   */
  async exploreConstraints(
    baseConfig: BenchmarkConfig,
    parameterRanges: {
      minWeight: [number, number],
      maxWeight: [number, number],
      targetReturn: [number, number],
    },
    samples: number = 20,
  ): Promise<SwarmInsights> {
    console.log('🔍 Exploring constraint space with swarm intelligence...');

    // Generate diverse constraint combinations
    const constraintVariations: PortfolioConstraints[] = [];

    for (let i = 0; i < samples; i++) {
      const t = i / (samples - 1);
      constraintVariations.push({
        minWeight: parameterRanges.minWeight[0] + t * (parameterRanges.minWeight[1] - parameterRanges.minWeight[0]),
        maxWeight: parameterRanges.maxWeight[0] + t * (parameterRanges.maxWeight[1] - parameterRanges.maxWeight[0]),
        targetReturn: parameterRanges.targetReturn[0] + t * (parameterRanges.targetReturn[1] - parameterRanges.targetReturn[0]),
      });
    }

    const config = {
      ...baseConfig,
      constraintVariations,
    };

    return await this.runBenchmark(config);
  }

  /**
   * Compare algorithm performance across different market regimes
   */
  async compareMarketRegimes(
    config: BenchmarkConfig,
    regimes: Array<{ name: string, volatilityMultiplier: number, returnMultiplier: number }>,
  ): Promise<Record<string, SwarmInsights>> {
    console.log('📈 Comparing performance across market regimes...');

    const results: Record<string, SwarmInsights> = {};

    for (const regime of regimes) {
      console.log(`\n🎯 Testing ${regime.name} regime...`);

      // Adjust asset parameters for regime
      const adjustedAssets = config.assets.map(asset => ({
        ...asset,
        expectedReturn: asset.expectedReturn * regime.returnMultiplier,
        volatility: asset.volatility * regime.volatilityMultiplier,
      }));

      const regimeConfig = {
        ...config,
        assets: adjustedAssets,
      };

      results[regime.name] = await this.runBenchmark(regimeConfig);
    }

    return results;
  }

  /**
   * Generate comprehensive benchmark report
   */
  generateReport(insights: SwarmInsights): string {
    let report = '\n📊 PORTFOLIO OPTIMIZATION BENCHMARK REPORT\n';
    report += '='.repeat(60) + '\n\n';

    report += `🏆 Best Algorithm: ${insights.bestAlgorithm}\n`;
    report += `📈 Best Sharpe Ratio: ${insights.bestResult.result.sharpeRatio.toFixed(4)}\n`;
    report += `⚠️  Risk Level: ${(insights.bestResult.result.risk * 100).toFixed(2)}%\n`;
    report += `💰 Expected Return: ${(insights.bestResult.result.expectedReturn * 100).toFixed(2)}%\n`;
    report += `🎯 Diversification Ratio: ${insights.bestResult.result.diversificationRatio.toFixed(4)}\n`;
    report += `⏱️  Execution Time: ${insights.bestResult.executionTime}ms\n\n`;

    report += '📊 Algorithm Rankings:\n';
    for (const ranking of insights.algorithmRankings) {
      report += `  ${ranking.algorithm.padEnd(20)} | Avg Sharpe: ${ranking.avgSharpe.toFixed(4)} | Avg Risk: ${(ranking.avgRisk * 100).toFixed(2)}%\n`;
    }

    report += '\n💡 Recommendations:\n';
    for (const rec of insights.recommendations) {
      report += `  • ${rec}\n`;
    }

    report += '\n' + '='.repeat(60) + '\n';

    return report;
  }
}

/**
 * Parallel Portfolio Explorer
 * Uses worker threads for CPU-intensive optimization
 */
export class ParallelPortfolioExplorer {
  private maxWorkers: number;

  constructor(maxWorkers: number = 4) {
    this.maxWorkers = maxWorkers;
  }

  /**
   * Run optimizations in parallel batches
   */
  async optimizeInParallel(
    configs: Array<{ algorithm: string, config: BenchmarkConfig, constraints: PortfolioConstraints }>,
  ): Promise<BenchmarkResult[]> {
    const results: BenchmarkResult[] = [];
    const swarm = new PortfolioOptimizationSwarm();

    // Process in batches
    for (let i = 0; i < configs.length; i += this.maxWorkers) {
      const batch = configs.slice(i, i + this.maxWorkers);
      const batchResults = await Promise.all(
        batch.map(({ algorithm, config, constraints }) =>
          swarm['runOptimization'](algorithm, config, constraints)
        )
      );
      results.push(...batchResults);
    }

    return results;
  }
}
