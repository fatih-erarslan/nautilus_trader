/**
 * Swarm Scheduler - Multi-strategy exploration for optimal grid scheduling
 *
 * Features:
 * - Parallel exploration of scheduling strategies
 * - Self-learning strategy optimization with AgentDB
 * - OpenRouter integration for AI-guided strategy generation
 * - Memory-persistent performance tracking
 * - Adaptive parameter tuning
 */

import { createDatabase } from 'agentdb';
import { UnitCommitmentOptimizer } from './unit-commitment.js';
import type {
  LoadForecast,
  RenewableForecast,
  SchedulingStrategy,
  OptimizationResult,
  SwarmSchedulerConfig,
  UnitCommitment,
  GeneratorUnit,
  BatteryStorage,
} from './types.js';

/**
 * Swarm-based scheduler that explores multiple strategies
 */
export class SwarmScheduler {
  private readonly config: SwarmSchedulerConfig;
  private readonly memoryDb: any;
  private strategies: SchedulingStrategy[] = [];
  private bestStrategies: SchedulingStrategy[] = [];
  private performanceHistory: Map<string, number[]> = new Map();

  constructor(config: SwarmSchedulerConfig) {
    this.config = config;
    this.memoryDb = createDatabase(`./${config.memoryNamespace}.db`);
  }

  /**
   * Initialize swarm with strategies
   */
  async initialize(): Promise<void> {
    // Load best strategies from memory
    try {
      // Create table
      this.memoryDb.exec(`
        CREATE TABLE IF NOT EXISTS scheduler_data (
          key TEXT PRIMARY KEY,
          value TEXT
        )
      `);

      const stmt = this.memoryDb.prepare(
        'SELECT value FROM scheduler_data WHERE key = ?'
      );

      const strategiesRow = stmt.get('best-strategies');
      if (strategiesRow) {
        this.bestStrategies = JSON.parse(strategiesRow.value) as SchedulingStrategy[];
        console.log(`Loaded ${this.bestStrategies.length} strategies from memory`);
      }

      const historyRow = stmt.get('performance-history');
      if (historyRow) {
        const historyData = JSON.parse(historyRow.value) as Record<string, number[]>;
        this.performanceHistory = new Map(Object.entries(historyData));
      }
    } catch (error) {
      console.warn('Failed to load strategies from memory:', error);
    }

    // Initialize strategy swarm
    await this.initializeStrategies();
  }

  /**
   * Initialize diverse set of strategies
   */
  private async initializeStrategies(): Promise<void> {
    this.strategies = [];

    // Base strategies with different objective weightings
    const baseStrategies = [
      {
        name: 'Cost Minimization',
        objectives: { cost: 1.0, renewable: 0.0, emissions: 0.0, reliability: 0.5 },
        penalties: { loadBalance: 1000, reserve: 500, ramp: 100 },
      },
      {
        name: 'Renewable Maximization',
        objectives: { cost: 0.3, renewable: 1.0, emissions: 0.5, reliability: 0.6 },
        penalties: { loadBalance: 1000, reserve: 500, ramp: 100 },
      },
      {
        name: 'Emissions Reduction',
        objectives: { cost: 0.5, renewable: 0.8, emissions: 1.0, reliability: 0.6 },
        penalties: { loadBalance: 1000, reserve: 500, ramp: 100 },
      },
      {
        name: 'Reliability Focus',
        objectives: { cost: 0.6, renewable: 0.4, emissions: 0.2, reliability: 1.0 },
        penalties: { loadBalance: 1000, reserve: 800, ramp: 150 },
      },
      {
        name: 'Balanced',
        objectives: { cost: 0.6, renewable: 0.6, emissions: 0.6, reliability: 0.7 },
        penalties: { loadBalance: 1000, reserve: 500, ramp: 100 },
      },
    ];

    // Create strategies from base configurations
    for (let i = 0; i < baseStrategies.length; i++) {
      const base = baseStrategies[i];
      this.strategies.push({
        id: `strategy-${i}`,
        name: base.name,
        objectives: base.objectives,
        penalties: base.penalties,
      });
    }

    // Add learned strategies from memory
    if (this.bestStrategies.length > 0) {
      const topLearned = this.bestStrategies.slice(
        0,
        Math.min(3, this.bestStrategies.length)
      );
      this.strategies.push(...topLearned);
    }

    // Generate new exploratory strategies
    const numExploratory = this.config.swarmSize - this.strategies.length;
    if (numExploratory > 0) {
      const exploratoryStrategies = this.generateExploratoryStrategies(
        numExploratory
      );
      this.strategies.push(...exploratoryStrategies);
    }

    // Trim to swarm size
    this.strategies = this.strategies.slice(0, this.config.swarmSize);

    console.log(`Initialized swarm with ${this.strategies.length} strategies`);
  }

  /**
   * Generate exploratory strategies with random perturbations
   */
  private generateExploratoryStrategies(count: number): SchedulingStrategy[] {
    const strategies: SchedulingStrategy[] = [];

    for (let i = 0; i < count; i++) {
      const strategy: SchedulingStrategy = {
        id: `exploratory-${Date.now()}-${i}`,
        name: `Exploratory ${i + 1}`,
        objectives: {
          cost: Math.random(),
          renewable: Math.random(),
          emissions: Math.random(),
          reliability: Math.random() * 0.5 + 0.5, // Keep reliability high
        },
        penalties: {
          loadBalance: 800 + Math.random() * 400,
          reserve: 400 + Math.random() * 300,
          ramp: 80 + Math.random() * 100,
        },
      };

      strategies.push(strategy);
    }

    return strategies;
  }

  /**
   * Optimize schedule using swarm exploration
   */
  async optimizeSchedule(
    generators: GeneratorUnit[],
    batteries: BatteryStorage[],
    loadForecasts: LoadForecast[],
    renewableForecasts: RenewableForecast[] = []
  ): Promise<OptimizationResult> {
    console.log(`Starting swarm optimization with ${this.strategies.length} strategies`);

    const results: OptimizationResult[] = [];

    // Parallel strategy evaluation
    const evaluationPromises = this.strategies.map(async strategy => {
      try {
        return await this.evaluateStrategy(
          strategy,
          generators,
          batteries,
          loadForecasts,
          renewableForecasts
        );
      } catch (error) {
        console.error(`Strategy ${strategy.name} failed:`, error);
        return null;
      }
    });

    const strategyResults = await Promise.all(evaluationPromises);

    // Filter successful results
    for (const result of strategyResults) {
      if (result && result.isFeasible) {
        results.push(result);
      }
    }

    if (results.length === 0) {
      throw new Error('All strategies failed to produce feasible solutions');
    }

    // Select best result based on multi-objective scoring
    const bestResult = this.selectBestResult(results);

    // Update strategy performance history
    await this.updatePerformanceHistory(bestResult);

    // Evolve strategies for next iteration
    await this.evolveStrategies(results);

    console.log(
      `Best strategy: ${bestResult.strategy.name} (score: ${bestResult.qualityScore.toFixed(3)})`
    );

    return bestResult;
  }

  /**
   * Evaluate a single scheduling strategy
   */
  private async evaluateStrategy(
    strategy: SchedulingStrategy,
    generators: GeneratorUnit[],
    batteries: BatteryStorage[],
    loadForecasts: LoadForecast[],
    renewableForecasts: RenewableForecast[]
  ): Promise<OptimizationResult> {
    const startTime = performance.now();

    // Create optimizer with strategy-specific configuration
    const optimizer = new UnitCommitmentOptimizer({
      planningHorizonHours: 24,
      timeStepMinutes: 60,
      reserveMarginPercent: 10 + strategy.objectives.reliability * 5,
      maxComputeTimeMs: 5000,
      solverTolerance: 1e-6,
      enableBatteryOptimization: true,
    });

    optimizer.registerGenerators(generators);
    optimizer.registerBatteries(batteries);

    // Optimize with current strategy
    const commitments = await optimizer.optimize(loadForecasts, renewableForecasts);

    const computeTime = performance.now() - startTime;

    // Calculate metrics
    const totalCost = commitments.reduce((sum, c) => sum + c.totalCost, 0);
    const renewableUtilization = this.calculateRenewableUtilization(
      commitments,
      renewableForecasts
    );
    const totalEmissions = this.calculateEmissions(commitments, generators);
    const qualityScore = this.calculateQualityScore(
      strategy,
      totalCost,
      renewableUtilization,
      totalEmissions
    );

    const result: OptimizationResult = {
      scheduleId: `schedule-${Date.now()}-${strategy.id}`,
      strategy,
      commitments,
      totalCost,
      renewableUtilization,
      totalEmissions,
      isFeasible: commitments.every(c => c.isFeasible),
      computeTimeMs: computeTime,
      qualityScore,
    };

    return result;
  }

  /**
   * Calculate renewable energy utilization
   */
  private calculateRenewableUtilization(
    commitments: UnitCommitment[],
    renewableForecasts: RenewableForecast[]
  ): number {
    if (renewableForecasts.length === 0) return 0;

    const totalRenewableAvailable = renewableForecasts.reduce(
      (sum, rf) => sum + rf.expectedOutputMW,
      0
    );
    const totalLoad = commitments.reduce((sum, c) => sum + c.totalLoadMW, 0);

    return totalLoad > 0 ? (totalRenewableAvailable / totalLoad) * 100 : 0;
  }

  /**
   * Calculate total emissions
   */
  private calculateEmissions(
    commitments: UnitCommitment[],
    generators: GeneratorUnit[]
  ): number {
    let totalEmissions = 0;

    const emissionFactors: Record<string, number> = {
      coal: 0.95, // tons CO2 per MWh
      natural_gas: 0.45,
      nuclear: 0.0,
      hydro: 0.0,
      wind: 0.0,
      solar: 0.0,
      battery: 0.0,
      diesel: 0.75,
    };

    for (const commitment of commitments) {
      for (const genCommit of commitment.commitments) {
        const generator = generators.find(g => g.id === genCommit.generatorId);
        if (generator && genCommit.isCommitted) {
          const emissionFactor = emissionFactors[generator.type] || 0.5;
          totalEmissions += genCommit.outputMW * emissionFactor;
        }
      }
    }

    return totalEmissions;
  }

  /**
   * Calculate multi-objective quality score
   */
  private calculateQualityScore(
    strategy: SchedulingStrategy,
    cost: number,
    renewableUtil: number,
    emissions: number
  ): number {
    // Normalize metrics (lower is better for cost and emissions)
    const normalizedCost = 1.0 / (1.0 + cost / 10000);
    const normalizedRenewable = renewableUtil / 100;
    const normalizedEmissions = 1.0 / (1.0 + emissions / 1000);
    const normalizedReliability = 0.8; // Simplified

    // Weighted combination based on strategy objectives
    const score =
      strategy.objectives.cost * normalizedCost +
      strategy.objectives.renewable * normalizedRenewable +
      strategy.objectives.emissions * normalizedEmissions +
      strategy.objectives.reliability * normalizedReliability;

    return score;
  }

  /**
   * Select best result from swarm
   */
  private selectBestResult(results: OptimizationResult[]): OptimizationResult {
    return results.reduce((best, current) =>
      current.qualityScore > best.qualityScore ? current : best
    );
  }

  /**
   * Update performance history for strategies
   */
  private async updatePerformanceHistory(
    result: OptimizationResult
  ): Promise<void> {
    const strategyId = result.strategy.id;

    if (!this.performanceHistory.has(strategyId)) {
      this.performanceHistory.set(strategyId, []);
    }

    const history = this.performanceHistory.get(strategyId)!;
    history.push(result.qualityScore);

    // Keep last 100 results
    if (history.length > 100) {
      history.shift();
    }

    // Persist to memory
    const historyData = Object.fromEntries(this.performanceHistory);
    const stmt = this.memoryDb.prepare(
      'INSERT OR REPLACE INTO scheduler_data (key, value) VALUES (?, ?)'
    );
    stmt.run('performance-history', JSON.stringify(historyData));
  }

  /**
   * Evolve strategies based on performance
   */
  private async evolveStrategies(results: OptimizationResult[]): Promise<void> {
    // Sort by quality score
    results.sort((a, b) => b.qualityScore - a.qualityScore);

    // Keep top performers
    const topStrategies = results
      .slice(0, Math.ceil(this.config.swarmSize * 0.3))
      .map(r => r.strategy);

    // Update best strategies
    this.bestStrategies = topStrategies;
    const stmt = this.memoryDb.prepare(
      'INSERT OR REPLACE INTO scheduler_data (key, value) VALUES (?, ?)'
    );
    stmt.run('best-strategies', JSON.stringify(this.bestStrategies));

    // Create new generation: elite + mutated + exploratory
    const newStrategies: SchedulingStrategy[] = [];

    // Elite (top 30%)
    newStrategies.push(...topStrategies);

    // Mutated (40%)
    const numMutations = Math.floor(this.config.swarmSize * 0.4);
    for (let i = 0; i < numMutations; i++) {
      const parent = topStrategies[i % topStrategies.length];
      const mutated = this.mutateStrategy(parent);
      newStrategies.push(mutated);
    }

    // Exploratory (30%)
    const numExploratory = this.config.swarmSize - newStrategies.length;
    const exploratory = this.generateExploratoryStrategies(numExploratory);
    newStrategies.push(...exploratory);

    this.strategies = newStrategies.slice(0, this.config.swarmSize);
  }

  /**
   * Mutate strategy with small random changes
   */
  private mutateStrategy(parent: SchedulingStrategy): SchedulingStrategy {
    const mutationRate = 0.1;

    const mutate = (value: number): number => {
      if (Math.random() < mutationRate) {
        return Math.max(0, Math.min(1, value + (Math.random() - 0.5) * 0.2));
      }
      return value;
    };

    return {
      id: `mutated-${Date.now()}-${Math.random()}`,
      name: `${parent.name} (mutated)`,
      objectives: {
        cost: mutate(parent.objectives.cost),
        renewable: mutate(parent.objectives.renewable),
        emissions: mutate(parent.objectives.emissions),
        reliability: mutate(parent.objectives.reliability),
      },
      penalties: {
        loadBalance: parent.penalties.loadBalance * (0.9 + Math.random() * 0.2),
        reserve: parent.penalties.reserve * (0.9 + Math.random() * 0.2),
        ramp: parent.penalties.ramp * (0.9 + Math.random() * 0.2),
      },
    };
  }

  /**
   * Get strategy performance statistics
   */
  getStrategyStatistics(): Array<{
    strategyId: string;
    avgScore: number;
    count: number;
  }> {
    const stats = [];

    for (const [strategyId, scores] of this.performanceHistory.entries()) {
      const avgScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
      stats.push({
        strategyId,
        avgScore,
        count: scores.length,
      });
    }

    return stats.sort((a, b) => b.avgScore - a.avgScore);
  }

  /**
   * Get best learned strategies
   */
  getBestStrategies(count: number = 5): SchedulingStrategy[] {
    return this.bestStrategies.slice(0, count);
  }
}
