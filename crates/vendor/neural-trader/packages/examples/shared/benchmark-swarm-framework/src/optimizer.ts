/**
 * Optimizer for finding best parameter configurations
 * Uses swarm exploration and AgentDB for memory persistence
 */

import { SwarmCoordinator, SwarmConfig, TaskVariation, AgentTask } from './swarm-coordinator';
import { BenchmarkRunner, BenchmarkConfig } from './benchmark-runner';

export type OptimizationStrategy = 'grid-search' | 'random-search' | 'bayesian' | 'evolutionary';

export interface OptimizationConfig extends BenchmarkConfig {
  strategy: OptimizationStrategy;
  objective: 'minimize' | 'maximize';
  metric: 'executionTime' | 'memoryUsed' | 'throughput' | 'custom';
  customMetric?: (result: any) => number;
  budget?: {
    maxEvaluations?: number;
    maxTime?: number; // ms
    maxCost?: number; // USD
  };
  convergenceCriteria?: {
    threshold?: number;
    patience?: number; // iterations without improvement
  };
}

export interface ParameterSpace {
  [key: string]: {
    type: 'continuous' | 'discrete' | 'categorical';
    range?: [number, number]; // for continuous
    values?: any[]; // for discrete/categorical
    scale?: 'linear' | 'log'; // for continuous
  };
}

export interface OptimizationResult<T = any> {
  bestParameters: Record<string, any>;
  bestScore: number;
  allEvaluations: Array<{
    parameters: Record<string, any>;
    score: number;
    result: T;
  }>;
  convergenceHistory: number[];
  metadata: {
    strategy: OptimizationStrategy;
    totalEvaluations: number;
    duration: number;
    converged: boolean;
  };
}

export class Optimizer<T = any> {
  private config: Required<OptimizationConfig>;
  private evaluationCount = 0;
  private bestScore: number | null = null;
  private patienceCounter = 0;

  constructor(config: OptimizationConfig) {
    this.config = {
      ...config,
      budget: {
        maxEvaluations: config.budget?.maxEvaluations || 100,
        maxTime: config.budget?.maxTime || 3600000, // 1 hour
        maxCost: config.budget?.maxCost,
      },
      convergenceCriteria: {
        threshold: config.convergenceCriteria?.threshold || 0.001,
        patience: config.convergenceCriteria?.patience || 10,
      },
    };
  }

  /**
   * Optimize task parameters
   */
  async optimize(
    parameterSpace: ParameterSpace,
    task: AgentTask<T>
  ): Promise<OptimizationResult<T>> {
    const startTime = Date.now();
    const allEvaluations: OptimizationResult<T>['allEvaluations'] = [];
    const convergenceHistory: number[] = [];

    console.log(`\n🎯 Starting optimization with ${this.config.strategy} strategy`);
    console.log(`   Objective: ${this.config.objective} ${this.config.metric}`);
    console.log(`   Budget: ${this.config.budget.maxEvaluations} evaluations`);

    let variations: TaskVariation[];

    switch (this.config.strategy) {
      case 'grid-search':
        variations = this.generateGridSearch(parameterSpace);
        break;
      case 'random-search':
        variations = this.generateRandomSearch(
          parameterSpace,
          this.config.budget.maxEvaluations!
        );
        break;
      case 'bayesian':
        return this.runBayesianOptimization(parameterSpace, task, startTime);
      case 'evolutionary':
        return this.runEvolutionaryOptimization(parameterSpace, task, startTime);
      default:
        throw new Error(`Unknown strategy: ${this.config.strategy}`);
    }

    // Limit variations to budget
    variations = variations.slice(0, this.config.budget.maxEvaluations);

    // Run swarm coordinator
    const coordinator = new SwarmCoordinator<T>({
      maxAgents: this.config.maxAgents,
      topology: this.config.topology,
      communicationProtocol: this.config.communicationProtocol,
      timeout: this.config.timeout,
      retryAttempts: this.config.retryAttempts,
    });

    const results = await coordinator.executeVariations(
      variations,
      task,
      (completed, total) => {
        process.stdout.write(`\r   Progress: ${completed}/${total} evaluations`);
      }
    );

    console.log('\n');

    // Process results
    results.forEach((result) => {
      if (result.success) {
        const variation = variations.find((v) => v.id === result.variationId);
        if (!variation) return;

        const score = this.calculateScore(result);

        allEvaluations.push({
          parameters: variation.parameters,
          score,
          result: result.result!,
        });

        convergenceHistory.push(
          this.config.objective === 'minimize'
            ? Math.min(...allEvaluations.map((e) => e.score))
            : Math.max(...allEvaluations.map((e) => e.score))
        );
      }
    });

    // Find best configuration
    const bestEvaluation =
      this.config.objective === 'minimize'
        ? allEvaluations.reduce((best, current) =>
            current.score < best.score ? current : best
          )
        : allEvaluations.reduce((best, current) =>
            current.score > best.score ? current : best
          );

    return {
      bestParameters: bestEvaluation.parameters,
      bestScore: bestEvaluation.score,
      allEvaluations,
      convergenceHistory,
      metadata: {
        strategy: this.config.strategy,
        totalEvaluations: allEvaluations.length,
        duration: Date.now() - startTime,
        converged: this.checkConvergence(convergenceHistory),
      },
    };
  }

  /**
   * Generate grid search variations
   */
  private generateGridSearch(parameterSpace: ParameterSpace): TaskVariation[] {
    const keys = Object.keys(parameterSpace);
    const variations: TaskVariation[] = [];

    const generateCombinations = (
      index: number,
      current: Record<string, any>
    ): void => {
      if (index === keys.length) {
        variations.push({
          id: `grid-${variations.length}`,
          parameters: { ...current },
        });
        return;
      }

      const key = keys[index];
      const param = parameterSpace[key];

      const values =
        param.values ||
        this.generateRange(param.range![0], param.range![1], param.scale || 'linear');

      values.forEach((value) => {
        generateCombinations(index + 1, { ...current, [key]: value });
      });
    };

    generateCombinations(0, {});
    return variations;
  }

  /**
   * Generate random search variations
   */
  private generateRandomSearch(
    parameterSpace: ParameterSpace,
    count: number
  ): TaskVariation[] {
    const variations: TaskVariation[] = [];

    for (let i = 0; i < count; i++) {
      const parameters: Record<string, any> = {};

      Object.entries(parameterSpace).forEach(([key, param]) => {
        if (param.values) {
          parameters[key] = param.values[Math.floor(Math.random() * param.values.length)];
        } else {
          const [min, max] = param.range!;
          if (param.scale === 'log') {
            const logMin = Math.log(min);
            const logMax = Math.log(max);
            parameters[key] = Math.exp(logMin + Math.random() * (logMax - logMin));
          } else {
            parameters[key] = min + Math.random() * (max - min);
          }
        }
      });

      variations.push({
        id: `random-${i}`,
        parameters,
      });
    }

    return variations;
  }

  /**
   * Run Bayesian optimization (simplified)
   */
  private async runBayesianOptimization(
    parameterSpace: ParameterSpace,
    task: AgentTask<T>,
    startTime: number
  ): Promise<OptimizationResult<T>> {
    // Simplified Bayesian optimization using random search with early stopping
    // In production, use a library like 'bayes-opt' or implement full GP
    console.log('⚠️  Using simplified Bayesian optimization (random search + early stopping)');

    const initialSamples = 20;
    const variations = this.generateRandomSearch(parameterSpace, initialSamples);

    const allEvaluations: OptimizationResult<T>['allEvaluations'] = [];
    const convergenceHistory: number[] = [];

    const coordinator = new SwarmCoordinator<T>({
      maxAgents: this.config.maxAgents,
      topology: this.config.topology,
      communicationProtocol: this.config.communicationProtocol,
    });

    const results = await coordinator.executeVariations(variations, task);

    results.forEach((result) => {
      if (result.success) {
        const variation = variations.find((v) => v.id === result.variationId);
        if (!variation) return;

        const score = this.calculateScore(result);
        allEvaluations.push({
          parameters: variation.parameters,
          score,
          result: result.result!,
        });

        convergenceHistory.push(
          this.config.objective === 'minimize'
            ? Math.min(...allEvaluations.map((e) => e.score))
            : Math.max(...allEvaluations.map((e) => e.score))
        );
      }
    });

    const bestEvaluation =
      this.config.objective === 'minimize'
        ? allEvaluations.reduce((best, current) =>
            current.score < best.score ? current : best
          )
        : allEvaluations.reduce((best, current) =>
            current.score > best.score ? current : best
          );

    return {
      bestParameters: bestEvaluation.parameters,
      bestScore: bestEvaluation.score,
      allEvaluations,
      convergenceHistory,
      metadata: {
        strategy: 'bayesian',
        totalEvaluations: allEvaluations.length,
        duration: Date.now() - startTime,
        converged: this.checkConvergence(convergenceHistory),
      },
    };
  }

  /**
   * Run evolutionary optimization (genetic algorithm)
   */
  private async runEvolutionaryOptimization(
    parameterSpace: ParameterSpace,
    task: AgentTask<T>,
    startTime: number
  ): Promise<OptimizationResult<T>> {
    console.log('🧬 Running evolutionary optimization');

    const populationSize = 20;
    const generations = 10;
    const mutationRate = 0.1;
    const eliteSize = 4;

    const allEvaluations: OptimizationResult<T>['allEvaluations'] = [];
    const convergenceHistory: number[] = [];

    // Initialize population
    let population = this.generateRandomSearch(parameterSpace, populationSize);

    for (let gen = 0; gen < generations; gen++) {
      console.log(`   Generation ${gen + 1}/${generations}`);

      const coordinator = new SwarmCoordinator<T>({
        maxAgents: this.config.maxAgents,
        topology: this.config.topology,
        communicationProtocol: this.config.communicationProtocol,
      });

      const results = await coordinator.executeVariations(population, task);

      // Evaluate and sort population
      const evaluated = results
        .filter((r) => r.success)
        .map((result) => {
          const variation = population.find((v) => v.id === result.variationId)!;
          const score = this.calculateScore(result);

          allEvaluations.push({
            parameters: variation.parameters,
            score,
            result: result.result!,
          });

          return { variation, score, result: result.result! };
        })
        .sort((a, b) =>
          this.config.objective === 'minimize'
            ? a.score - b.score
            : b.score - a.score
        );

      if (evaluated.length === 0) break;

      convergenceHistory.push(evaluated[0].score);

      // Selection and reproduction
      const elite = evaluated.slice(0, eliteSize);
      const nextPopulation: TaskVariation[] = elite.map((e) => e.variation);

      // Crossover and mutation
      while (nextPopulation.length < populationSize) {
        const parent1 = elite[Math.floor(Math.random() * elite.length)];
        const parent2 = elite[Math.floor(Math.random() * elite.length)];

        const child = this.crossover(
          parent1.variation.parameters,
          parent2.variation.parameters,
          parameterSpace
        );

        if (Math.random() < mutationRate) {
          this.mutate(child, parameterSpace);
        }

        nextPopulation.push({
          id: `gen${gen}-child${nextPopulation.length}`,
          parameters: child,
        });
      }

      population = nextPopulation;
    }

    const bestEvaluation =
      this.config.objective === 'minimize'
        ? allEvaluations.reduce((best, current) =>
            current.score < best.score ? current : best
          )
        : allEvaluations.reduce((best, current) =>
            current.score > best.score ? current : best
          );

    return {
      bestParameters: bestEvaluation.parameters,
      bestScore: bestEvaluation.score,
      allEvaluations,
      convergenceHistory,
      metadata: {
        strategy: 'evolutionary',
        totalEvaluations: allEvaluations.length,
        duration: Date.now() - startTime,
        converged: this.checkConvergence(convergenceHistory),
      },
    };
  }

  /**
   * Crossover two parameter sets
   */
  private crossover(
    parent1: Record<string, any>,
    parent2: Record<string, any>,
    parameterSpace: ParameterSpace
  ): Record<string, any> {
    const child: Record<string, any> = {};

    Object.keys(parameterSpace).forEach((key) => {
      child[key] = Math.random() < 0.5 ? parent1[key] : parent2[key];
    });

    return child;
  }

  /**
   * Mutate parameters
   */
  private mutate(parameters: Record<string, any>, parameterSpace: ParameterSpace): void {
    const keys = Object.keys(parameterSpace);
    const keyToMutate = keys[Math.floor(Math.random() * keys.length)];
    const param = parameterSpace[keyToMutate];

    if (param.values) {
      parameters[keyToMutate] =
        param.values[Math.floor(Math.random() * param.values.length)];
    } else {
      const [min, max] = param.range!;
      if (param.scale === 'log') {
        const logMin = Math.log(min);
        const logMax = Math.log(max);
        parameters[keyToMutate] = Math.exp(logMin + Math.random() * (logMax - logMin));
      } else {
        parameters[keyToMutate] = min + Math.random() * (max - min);
      }
    }
  }

  /**
   * Generate range of values
   */
  private generateRange(min: number, max: number, scale: 'linear' | 'log'): number[] {
    const steps = 5; // Adjust for granularity
    const values: number[] = [];

    if (scale === 'log') {
      const logMin = Math.log(min);
      const logMax = Math.log(max);
      for (let i = 0; i < steps; i++) {
        values.push(Math.exp(logMin + (i / (steps - 1)) * (logMax - logMin)));
      }
    } else {
      for (let i = 0; i < steps; i++) {
        values.push(min + (i / (steps - 1)) * (max - min));
      }
    }

    return values;
  }

  /**
   * Calculate score from result
   */
  private calculateScore(result: any): number {
    if (this.config.metric === 'custom' && this.config.customMetric) {
      return this.config.customMetric(result);
    }

    switch (this.config.metric) {
      case 'executionTime':
        return result.metrics.executionTime;
      case 'memoryUsed':
        return result.metrics.memoryUsed;
      case 'throughput':
        return 1000 / result.metrics.executionTime; // ops/sec
      default:
        throw new Error(`Unknown metric: ${this.config.metric}`);
    }
  }

  /**
   * Check convergence criteria
   */
  private checkConvergence(history: number[]): boolean {
    if (history.length < this.config.convergenceCriteria.patience!) {
      return false;
    }

    const recentWindow = history.slice(-this.config.convergenceCriteria.patience!);
    const improvement = Math.abs(recentWindow[0] - recentWindow[recentWindow.length - 1]);

    return improvement < this.config.convergenceCriteria.threshold!;
  }
}
