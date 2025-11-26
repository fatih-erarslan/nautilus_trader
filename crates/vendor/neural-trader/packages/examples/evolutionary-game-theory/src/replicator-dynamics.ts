/**
 * Replicator Dynamics Simulation
 *
 * The replicator equation describes how strategy frequencies evolve:
 * dx_i/dt = x_i * (f_i - f_avg)
 *
 * where:
 * - x_i is frequency of strategy i
 * - f_i is fitness of strategy i
 * - f_avg is average fitness of population
 */

import type { Game, PopulationState } from './types.js';
import { calculateFitnessValues } from './games.js';

/**
 * Replicator dynamics simulator
 */
export class ReplicatorDynamics {
  private game: Game;
  private currentState: PopulationState;
  private history: PopulationState[] = [];

  constructor(game: Game, initialPopulation?: number[]) {
    this.game = game;

    // Initialize with uniform distribution if not provided
    const initial = initialPopulation ||
      new Array(game.numStrategies).fill(1 / game.numStrategies);

    // Normalize to ensure sum = 1
    const sum = initial.reduce((a, b) => a + b, 0);
    const normalized = initial.map((x) => x / sum);

    this.currentState = {
      frequencies: normalized,
      generation: 0,
      timestamp: Date.now(),
    };

    this.history.push({ ...this.currentState });
  }

  /**
   * Calculate fitness values for current population
   */
  private calculateFitness(): number[] {
    return calculateFitnessValues(this.game, this.currentState.frequencies);
  }

  /**
   * Calculate average fitness of population
   */
  private calculateAverageFitness(fitnessValues: number[]): number {
    let avgFitness = 0;
    for (let i = 0; i < this.game.numStrategies; i++) {
      avgFitness += this.currentState.frequencies[i] * fitnessValues[i];
    }
    return avgFitness;
  }

  /**
   * Perform one step of replicator dynamics
   * @param dt Time step size
   */
  step(dt: number = 0.01): PopulationState {
    const fitness = this.calculateFitness();
    const avgFitness = this.calculateAverageFitness(fitness);

    // Update frequencies using replicator equation
    const newFrequencies = this.currentState.frequencies.map((x_i, i) => {
      const f_i = fitness[i];
      // dx_i/dt = x_i * (f_i - f_avg)
      const delta = x_i * (f_i - avgFitness) * dt;
      return Math.max(0, x_i + delta); // Ensure non-negative
    });

    // Normalize to ensure sum = 1
    const sum = newFrequencies.reduce((a, b) => a + b, 0);
    const normalized = sum > 0
      ? newFrequencies.map((x) => x / sum)
      : this.currentState.frequencies; // Keep old if all died

    this.currentState = {
      frequencies: normalized,
      generation: this.currentState.generation + 1,
      timestamp: Date.now(),
      fitnessValues: fitness,
      averageFitness: avgFitness,
    };

    this.history.push({ ...this.currentState });
    return this.currentState;
  }

  /**
   * Simulate for multiple steps
   * @param steps Number of steps
   * @param dt Time step size
   */
  simulate(steps: number, dt: number = 0.01): PopulationState[] {
    const results: PopulationState[] = [];
    for (let i = 0; i < steps; i++) {
      results.push(this.step(dt));
    }
    return results;
  }

  /**
   * Simulate until convergence or max steps
   * @param threshold Convergence threshold
   * @param maxSteps Maximum steps
   * @param dt Time step size
   */
  simulateUntilConvergence(
    threshold: number = 1e-6,
    maxSteps: number = 10000,
    dt: number = 0.01
  ): PopulationState {
    for (let i = 0; i < maxSteps; i++) {
      const prevState = { ...this.currentState };
      this.step(dt);

      // Check convergence
      const maxChange = Math.max(
        ...this.currentState.frequencies.map((x, j) =>
          Math.abs(x - prevState.frequencies[j])
        )
      );

      if (maxChange < threshold) {
        break;
      }
    }

    return this.currentState;
  }

  /**
   * Check if current state is at fixed point
   */
  isFixedPoint(threshold: number = 1e-6): boolean {
    const fitness = this.calculateFitness();
    const avgFitness = this.calculateAverageFitness(fitness);

    // At fixed point, all active strategies have equal fitness
    for (let i = 0; i < this.game.numStrategies; i++) {
      if (this.currentState.frequencies[i] > threshold) {
        if (Math.abs(fitness[i] - avgFitness) > threshold) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Calculate Shannon diversity index
   */
  calculateDiversity(): number {
    let diversity = 0;
    for (const freq of this.currentState.frequencies) {
      if (freq > 0) {
        diversity -= freq * Math.log(freq);
      }
    }
    return diversity;
  }

  /**
   * Get current state
   */
  getState(): PopulationState {
    return { ...this.currentState };
  }

  /**
   * Get simulation history
   */
  getHistory(): PopulationState[] {
    return [...this.history];
  }

  /**
   * Reset to initial state
   */
  reset(initialPopulation?: number[]): void {
    const initial = initialPopulation ||
      new Array(this.game.numStrategies).fill(1 / this.game.numStrategies);

    const sum = initial.reduce((a, b) => a + b, 0);
    const normalized = initial.map((x) => x / sum);

    this.currentState = {
      frequencies: normalized,
      generation: 0,
      timestamp: Date.now(),
    };

    this.history = [{ ...this.currentState }];
  }

  /**
   * Find all fixed points in the simplex (discrete approximation)
   */
  findFixedPoints(
    resolution: number = 10,
    threshold: number = 1e-4
  ): number[][] {
    const fixedPoints: number[][] = [];

    // Generate all points in the simplex
    const points = this.generateSimplexPoints(resolution);

    for (const point of points) {
      this.reset(point);
      const initial = [...point];

      // Simulate a few steps
      this.simulate(100, 0.01);

      // Check if we're at a fixed point
      if (this.isFixedPoint(threshold)) {
        // Check if this is a new fixed point
        const isNew = !fixedPoints.some((fp) =>
          fp.every((x, i) => Math.abs(x - this.currentState.frequencies[i]) < threshold)
        );

        if (isNew) {
          fixedPoints.push([...this.currentState.frequencies]);
        }
      }
    }

    return fixedPoints;
  }

  /**
   * Generate points on the simplex for scanning
   */
  private generateSimplexPoints(resolution: number): number[][] {
    const points: number[][] = [];

    const generateRecursive = (
      remaining: number,
      depth: number,
      current: number[]
    ): void => {
      if (depth === this.game.numStrategies - 1) {
        // Last dimension is determined
        points.push([...current, remaining]);
        return;
      }

      // Try all possible values for this dimension
      for (let i = 0; i <= remaining * resolution; i++) {
        const value = i / resolution;
        generateRecursive(remaining - value, depth + 1, [...current, value]);
      }
    };

    generateRecursive(1.0, 0, []);
    return points;
  }

  /**
   * Calculate phase portrait velocities at a point
   */
  getVelocity(population: number[]): number[] {
    const fitness = calculateFitnessValues(this.game, population);
    let avgFitness = 0;
    for (let i = 0; i < this.game.numStrategies; i++) {
      avgFitness += population[i] * fitness[i];
    }

    return population.map((x_i, i) => x_i * (fitness[i] - avgFitness));
  }

  /**
   * Export state for visualization
   */
  exportForVisualization(): {
    game: Game;
    states: PopulationState[];
    fixedPoints?: number[][];
  } {
    return {
      game: this.game,
      states: this.history,
      fixedPoints: this.game.numStrategies <= 3
        ? this.findFixedPoints(5, 1e-3)
        : undefined,
    };
  }
}

/**
 * Multi-population replicator dynamics
 * Useful for coevolution and multi-species games
 */
export class MultiPopulationDynamics {
  private populations: ReplicatorDynamics[];
  private interactionMatrix: number[][]; // How populations interact

  constructor(games: Game[], interactionMatrix?: number[][]) {
    this.populations = games.map((game) => new ReplicatorDynamics(game));

    // Default: all populations interact equally
    this.interactionMatrix = interactionMatrix ||
      games.map(() => games.map(() => 1.0));
  }

  /**
   * Step all populations simultaneously
   */
  step(dt: number = 0.01): PopulationState[] {
    return this.populations.map((pop) => pop.step(dt));
  }

  /**
   * Simulate all populations
   */
  simulate(steps: number, dt: number = 0.01): PopulationState[][] {
    const results: PopulationState[][] = Array(this.populations.length)
      .fill(null)
      .map(() => []);

    for (let i = 0; i < steps; i++) {
      const states = this.step(dt);
      states.forEach((state, j) => results[j].push(state));
    }

    return results;
  }

  /**
   * Get all population states
   */
  getStates(): PopulationState[] {
    return this.populations.map((pop) => pop.getState());
  }

  /**
   * Calculate cross-population diversity
   */
  calculateCrossDiversity(): number {
    const states = this.getStates();
    let totalDiversity = 0;

    for (let i = 0; i < states.length; i++) {
      for (let j = i + 1; j < states.length; j++) {
        // Calculate distance between populations
        let distance = 0;
        const minLen = Math.min(
          states[i].frequencies.length,
          states[j].frequencies.length
        );

        for (let k = 0; k < minLen; k++) {
          distance += Math.abs(
            states[i].frequencies[k] - states[j].frequencies[k]
          );
        }

        totalDiversity += distance;
      }
    }

    return totalDiversity;
  }
}
