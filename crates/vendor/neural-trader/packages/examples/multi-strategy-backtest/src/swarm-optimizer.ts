/**
 * Particle Swarm Optimization for strategy parameter tuning
 * Implements PSO with adaptive inertia and constraint handling
 */

import {
  SwarmConfig,
  SwarmParticle,
  OptimizationResult,
  BacktestConfig,
  MarketData,
  StrategyPerformance
} from './types';
import { Backtester } from './backtester';
import { Strategy } from './types';

export type ObjectiveFunction = (parameters: Record<string, number>) => Promise<number>;

export class SwarmOptimizer {
  private config: SwarmConfig;
  private particles: SwarmParticle[] = [];
  private globalBestPosition: Record<string, number> = {};
  private globalBestScore: number = -Infinity;
  private convergenceHistory: number[] = [];
  private evaluationCount: number = 0;

  constructor(config: Partial<SwarmConfig> = {}) {
    this.config = {
      particleCount: 30,
      maxIterations: 100,
      inertia: 0.7,
      cognitiveWeight: 1.5,
      socialWeight: 1.5,
      bounds: {},
      ...config
    };
  }

  /**
   * Optimize strategy parameters using PSO
   */
  async optimize(
    objectiveFunction: ObjectiveFunction,
    bounds: Record<string, [number, number]>
  ): Promise<OptimizationResult> {
    console.log('\n🐝 Starting Particle Swarm Optimization...');
    console.log(`📊 Particles: ${this.config.particleCount}, Iterations: ${this.config.maxIterations}`);

    const startTime = Date.now();
    this.config.bounds = bounds;

    // Initialize swarm
    this.initializeSwarm();

    // Evaluate initial positions
    await this.evaluateSwarm(objectiveFunction);

    // Main PSO loop
    for (let iter = 0; iter < this.config.maxIterations; iter++) {
      // Update particles
      for (const particle of this.particles) {
        this.updateParticle(particle);
      }

      // Evaluate new positions
      await this.evaluateSwarm(objectiveFunction);

      // Track convergence
      this.convergenceHistory.push(this.globalBestScore);

      // Log progress
      if ((iter + 1) % 10 === 0) {
        console.log(`🔄 Iteration ${iter + 1}/${this.config.maxIterations}, Best: ${this.globalBestScore.toFixed(4)}`);
      }

      // Early stopping if converged
      if (this.hasConverged()) {
        console.log(`✅ Converged at iteration ${iter + 1}`);
        break;
      }

      // Adaptive inertia
      this.config.inertia = this.calculateAdaptiveInertia(iter);
    }

    const timeElapsed = Date.now() - startTime;

    console.log(`\n🎯 Optimization Complete!`);
    console.log(`⏱️  Time: ${(timeElapsed / 1000).toFixed(2)}s`);
    console.log(`📈 Evaluations: ${this.evaluationCount}`);
    console.log(`🏆 Best Score: ${this.globalBestScore.toFixed(4)}`);
    console.log(`📊 Best Parameters:`, this.globalBestPosition);

    return {
      bestParameters: this.globalBestPosition,
      bestScore: this.globalBestScore,
      convergenceHistory: this.convergenceHistory,
      evaluations: this.evaluationCount,
      timeElapsed
    };
  }

  /**
   * Optimize multiple strategies in parallel using swarm
   */
  async optimizeMultipleStrategies(
    backtestConfig: BacktestConfig,
    marketData: MarketData[],
    strategies: Strategy[]
  ): Promise<Map<string, OptimizationResult>> {
    console.log('\n🚀 Multi-Strategy Swarm Optimization...');

    const results = new Map<string, OptimizationResult>();

    // Optimize each strategy concurrently
    const optimizations = strategies.map(async (strategy) => {
      console.log(`\n🎯 Optimizing ${strategy.name}...`);

      // Define objective function for this strategy
      const objective = async (params: Record<string, number>): Promise<number> => {
        // Update strategy parameters
        strategy.updateParameters(params);

        // Run backtest
        const backtester = new Backtester(backtestConfig, [strategy]);
        const performances = await backtester.runBacktest(marketData);

        // Return negative Sharpe ratio (PSO minimizes)
        const performance = performances.find(p => p.strategyName === strategy.name);
        return performance ? performance.sharpeRatio : -Infinity;
      };

      // Get parameter bounds from strategy
      const bounds = this.getStrategyBounds(strategy);

      // Run optimization
      const result = await this.optimize(objective, bounds);
      results.set(strategy.name, result);

      return result;
    });

    await Promise.all(optimizations);

    console.log('\n✅ All strategies optimized!');
    return results;
  }

  /**
   * Initialize particle swarm
   */
  private initializeSwarm(): void {
    this.particles = [];
    const paramNames = Object.keys(this.config.bounds);

    for (let i = 0; i < this.config.particleCount; i++) {
      const position: Record<string, number> = {};
      const velocity: Record<string, number> = {};

      for (const param of paramNames) {
        const [min, max] = this.config.bounds[param];
        position[param] = min + Math.random() * (max - min);
        velocity[param] = (Math.random() - 0.5) * (max - min) * 0.1;
      }

      const particle: SwarmParticle = {
        id: `particle-${i}`,
        position,
        velocity,
        bestPosition: { ...position },
        bestScore: -Infinity,
        currentScore: -Infinity
      };

      this.particles.push(particle);
    }
  }

  /**
   * Evaluate all particles
   */
  private async evaluateSwarm(objective: ObjectiveFunction): Promise<void> {
    const evaluations = this.particles.map(async (particle) => {
      const score = await objective(particle.position);
      this.evaluationCount++;

      particle.currentScore = score;

      // Update personal best
      if (score > particle.bestScore) {
        particle.bestScore = score;
        particle.bestPosition = { ...particle.position };
      }

      // Update global best
      if (score > this.globalBestScore) {
        this.globalBestScore = score;
        this.globalBestPosition = { ...particle.position };
      }

      return score;
    });

    await Promise.all(evaluations);
  }

  /**
   * Update particle position and velocity
   */
  private updateParticle(particle: SwarmParticle): void {
    const paramNames = Object.keys(this.config.bounds);

    for (const param of paramNames) {
      // Random factors
      const r1 = Math.random();
      const r2 = Math.random();

      // Velocity update
      const cognitive = this.config.cognitiveWeight * r1 *
        (particle.bestPosition[param] - particle.position[param]);
      const social = this.config.socialWeight * r2 *
        (this.globalBestPosition[param] - particle.position[param]);

      particle.velocity[param] =
        this.config.inertia * particle.velocity[param] + cognitive + social;

      // Velocity clamping
      const [min, max] = this.config.bounds[param];
      const maxVelocity = (max - min) * 0.2;
      particle.velocity[param] = Math.max(-maxVelocity,
        Math.min(maxVelocity, particle.velocity[param]));

      // Position update
      particle.position[param] += particle.velocity[param];

      // Boundary handling
      particle.position[param] = Math.max(min, Math.min(max, particle.position[param]));
    }
  }

  /**
   * Calculate adaptive inertia weight
   */
  private calculateAdaptiveInertia(iteration: number): number {
    const maxInertia = 0.9;
    const minInertia = 0.4;
    return maxInertia - ((maxInertia - minInertia) * iteration) / this.config.maxIterations;
  }

  /**
   * Check if swarm has converged
   */
  private hasConverged(): boolean {
    if (this.convergenceHistory.length < 10) return false;

    const recent = this.convergenceHistory.slice(-10);
    const variance = this.calculateVariance(recent);

    return variance < 1e-6;
  }

  /**
   * Calculate variance of array
   */
  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
  }

  /**
   * Get parameter bounds for a strategy
   */
  private getStrategyBounds(strategy: Strategy): Record<string, [number, number]> {
    const params = strategy.getParameters();
    const bounds: Record<string, [number, number]> = {};

    // Default bounds based on parameter type
    for (const [key, value] of Object.entries(params)) {
      if (typeof value === 'number') {
        if (key.includes('period') || key.includes('window')) {
          bounds[key] = [5, 200];
        } else if (key.includes('threshold') || key.includes('ratio')) {
          bounds[key] = [0, 1];
        } else if (key.includes('multiplier') || key.includes('factor')) {
          bounds[key] = [0.5, 3.0];
        } else {
          bounds[key] = [value * 0.5, value * 2.0];
        }
      }
    }

    return bounds;
  }

  /**
   * Get swarm statistics
   */
  getStats(): {
    particleCount: number;
    iterations: number;
    evaluations: number;
    globalBest: number;
    convergenceHistory: number[];
  } {
    return {
      particleCount: this.particles.length,
      iterations: this.convergenceHistory.length,
      evaluations: this.evaluationCount,
      globalBest: this.globalBestScore,
      convergenceHistory: this.convergenceHistory
    };
  }

  /**
   * Reset optimizer
   */
  reset(): void {
    this.particles = [];
    this.globalBestPosition = {};
    this.globalBestScore = -Infinity;
    this.convergenceHistory = [];
    this.evaluationCount = 0;
  }
}
