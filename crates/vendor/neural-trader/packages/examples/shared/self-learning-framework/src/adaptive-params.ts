/**
 * Adaptive parameter tuning based on performance feedback
 * Automatically adjusts parameters to optimize outcomes
 */

import { ExperienceReplay, Experience } from './experience-replay';
import { PatternLearner, Pattern } from './pattern-learner';

export interface Parameter {
  name: string;
  type: 'continuous' | 'discrete' | 'categorical';
  range?: [number, number];
  values?: any[];
  default: any;
  current: any;
}

export interface AdaptationConfig {
  learningRate: number;
  adaptationInterval: number; // experiences between adaptations
  explorationRate: number; // epsilon for epsilon-greedy
  decayRate: number; // decay for exploration rate
  minExplorationRate: number;
}

export interface PerformanceMetrics {
  avgReward: number;
  successRate: number;
  recentReward: number;
  trend: 'improving' | 'stable' | 'declining';
}

export class AdaptiveParameters {
  private parameters: Map<string, Parameter> = new Map();
  private replay: ExperienceReplay;
  private learner: PatternLearner;
  private config: Required<AdaptationConfig>;
  private experiencesSinceAdaptation: number = 0;
  private performanceHistory: number[] = [];

  constructor(
    replay: ExperienceReplay,
    learner: PatternLearner,
    config: AdaptationConfig
  ) {
    this.replay = replay;
    this.learner = learner;
    this.config = {
      learningRate: config.learningRate,
      adaptationInterval: config.adaptationInterval,
      explorationRate: config.explorationRate,
      decayRate: config.decayRate,
      minExplorationRate: config.minExplorationRate,
    };
  }

  /**
   * Register a parameter for adaptive tuning
   */
  registerParameter(parameter: Parameter): void {
    this.parameters.set(parameter.name, { ...parameter });
  }

  /**
   * Register multiple parameters
   */
  registerParameters(parameters: Parameter[]): void {
    parameters.forEach((param) => this.registerParameter(param));
  }

  /**
   * Get current parameter value
   */
  getParameter(name: string): any {
    const param = this.parameters.get(name);
    if (!param) {
      throw new Error(`Parameter ${name} not registered`);
    }
    return param.current;
  }

  /**
   * Get all parameters
   */
  getAllParameters(): Record<string, any> {
    const result: Record<string, any> = {};
    this.parameters.forEach((param, name) => {
      result[name] = param.current;
    });
    return result;
  }

  /**
   * Record experience and adapt if needed
   */
  async recordExperience(experience: Experience): Promise<boolean> {
    this.experiencesSinceAdaptation++;
    this.performanceHistory.push(experience.reward);

    // Keep only recent history (last 100)
    if (this.performanceHistory.length > 100) {
      this.performanceHistory.shift();
    }

    // Check if adaptation is needed
    if (this.experiencesSinceAdaptation >= this.config.adaptationInterval) {
      await this.adapt();
      this.experiencesSinceAdaptation = 0;
      return true;
    }

    return false;
  }

  /**
   * Adapt parameters based on performance
   */
  async adapt(): Promise<void> {
    console.log('🔄 Adapting parameters...');

    const metrics = await this.getPerformanceMetrics();

    console.log(`   Current performance: ${metrics.avgReward.toFixed(3)} (${metrics.trend})`);

    // Get best patterns
    const topPatterns = this.learner.getBestRewardPatterns(5);

    if (topPatterns.length === 0) {
      console.log('   ⚠️  No patterns available for adaptation');
      return;
    }

    // Adapt each parameter based on successful patterns
    for (const [name, param] of this.parameters.entries()) {
      const shouldExplore = Math.random() < this.config.explorationRate;

      if (shouldExplore) {
        // Exploration: try random value
        this.exploreParameter(param);
        console.log(`   🔍 Exploring ${name}: ${param.current}`);
      } else {
        // Exploitation: use value from best pattern
        this.exploitParameter(param, topPatterns, metrics);
        console.log(`   ✅ Exploiting ${name}: ${param.current}`);
      }
    }

    // Decay exploration rate
    this.config.explorationRate = Math.max(
      this.config.minExplorationRate,
      this.config.explorationRate * this.config.decayRate
    );

    console.log(`   Exploration rate: ${this.config.explorationRate.toFixed(3)}`);
  }

  /**
   * Get performance metrics
   */
  private async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const recentExperiences = await this.replay.getAll();

    if (recentExperiences.length === 0) {
      return {
        avgReward: 0,
        successRate: 0,
        recentReward: 0,
        trend: 'stable',
      };
    }

    const avgReward =
      recentExperiences.reduce((sum, e) => sum + e.reward, 0) /
      recentExperiences.length;

    const successRate =
      recentExperiences.filter((e) => e.reward > 0).length /
      recentExperiences.length;

    const recentReward =
      this.performanceHistory.slice(-10).reduce((sum, r) => sum + r, 0) / 10;

    // Determine trend
    let trend: PerformanceMetrics['trend'] = 'stable';
    if (this.performanceHistory.length >= 20) {
      const firstHalf =
        this.performanceHistory
          .slice(0, Math.floor(this.performanceHistory.length / 2))
          .reduce((sum, r) => sum + r, 0) /
        Math.floor(this.performanceHistory.length / 2);

      const secondHalf =
        this.performanceHistory
          .slice(Math.floor(this.performanceHistory.length / 2))
          .reduce((sum, r) => sum + r, 0) /
        Math.ceil(this.performanceHistory.length / 2);

      const improvement = (secondHalf - firstHalf) / Math.abs(firstHalf);

      if (improvement > 0.1) {
        trend = 'improving';
      } else if (improvement < -0.1) {
        trend = 'declining';
      }
    }

    return {
      avgReward,
      successRate,
      recentReward,
      trend,
    };
  }

  /**
   * Explore: try random parameter value
   */
  private exploreParameter(param: Parameter): void {
    switch (param.type) {
      case 'continuous':
        if (param.range) {
          const [min, max] = param.range;
          param.current = min + Math.random() * (max - min);
        }
        break;

      case 'discrete':
      case 'categorical':
        if (param.values) {
          const index = Math.floor(Math.random() * param.values.length);
          param.current = param.values[index];
        }
        break;
    }
  }

  /**
   * Exploit: use value from successful patterns
   */
  private exploitParameter(
    param: Parameter,
    topPatterns: Pattern[],
    metrics: PerformanceMetrics
  ): void {
    // Extract parameter value from best pattern
    // This is simplified - in production, extract actual parameter values from pattern templates
    const bestPattern = topPatterns[0];

    if (metrics.trend === 'declining') {
      // If performance is declining, revert to default or try something new
      param.current = param.default;
    } else {
      // Move current value slightly towards best performing value
      // This is a placeholder - actual implementation depends on pattern structure
      switch (param.type) {
        case 'continuous':
          if (param.range) {
            const [min, max] = param.range;
            // Move slightly in a random direction
            const delta = ((max - min) * this.config.learningRate) * (Math.random() - 0.5);
            param.current = Math.max(min, Math.min(max, param.current + delta));
          }
          break;

        case 'discrete':
        case 'categorical':
          // Keep current value if performance is good
          if (metrics.trend !== 'improving' && param.values) {
            const currentIndex = param.values.indexOf(param.current);
            const neighbors = [
              param.values[Math.max(0, currentIndex - 1)],
              param.values[Math.min(param.values.length - 1, currentIndex + 1)],
            ];
            param.current = neighbors[Math.floor(Math.random() * neighbors.length)];
          }
          break;
      }
    }
  }

  /**
   * Reset parameter to default
   */
  resetParameter(name: string): void {
    const param = this.parameters.get(name);
    if (param) {
      param.current = param.default;
    }
  }

  /**
   * Reset all parameters to defaults
   */
  resetAllParameters(): void {
    this.parameters.forEach((param) => {
      param.current = param.default;
    });
  }

  /**
   * Set exploration rate
   */
  setExplorationRate(rate: number): void {
    this.config.explorationRate = Math.max(
      this.config.minExplorationRate,
      Math.min(1.0, rate)
    );
  }

  /**
   * Get exploration rate
   */
  getExplorationRate(): number {
    return this.config.explorationRate;
  }

  /**
   * Export configuration
   */
  export(): {
    parameters: Record<string, any>;
    config: AdaptationConfig;
    performanceHistory: number[];
  } {
    const parameters: Record<string, any> = {};
    this.parameters.forEach((param, name) => {
      parameters[name] = {
        ...param,
      };
    });

    return {
      parameters,
      config: this.config,
      performanceHistory: [...this.performanceHistory],
    };
  }

  /**
   * Import configuration
   */
  import(data: ReturnType<AdaptiveParameters['export']>): void {
    this.parameters.clear();
    Object.entries(data.parameters).forEach(([name, param]) => {
      this.parameters.set(name, param as Parameter);
    });

    Object.assign(this.config, data.config);
    this.performanceHistory = [...data.performanceHistory];
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalParameters: number;
    explorationRate: number;
    avgPerformance: number;
    performanceTrend: 'improving' | 'stable' | 'declining';
  } {
    const avgPerformance =
      this.performanceHistory.length > 0
        ? this.performanceHistory.reduce((sum, r) => sum + r, 0) /
          this.performanceHistory.length
        : 0;

    let trend: 'improving' | 'stable' | 'declining' = 'stable';
    if (this.performanceHistory.length >= 20) {
      const firstHalf =
        this.performanceHistory
          .slice(0, Math.floor(this.performanceHistory.length / 2))
          .reduce((sum, r) => sum + r, 0) /
        Math.floor(this.performanceHistory.length / 2);

      const secondHalf =
        this.performanceHistory
          .slice(Math.floor(this.performanceHistory.length / 2))
          .reduce((sum, r) => sum + r, 0) /
        Math.ceil(this.performanceHistory.length / 2);

      const improvement = (secondHalf - firstHalf) / Math.abs(firstHalf);

      if (improvement > 0.1) {
        trend = 'improving';
      } else if (improvement < -0.1) {
        trend = 'declining';
      }
    }

    return {
      totalParameters: this.parameters.size,
      explorationRate: this.config.explorationRate,
      avgPerformance,
      performanceTrend: trend,
    };
  }
}
