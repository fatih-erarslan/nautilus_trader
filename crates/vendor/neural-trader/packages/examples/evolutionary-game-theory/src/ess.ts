/**
 * Evolutionarily Stable Strategy (ESS) Analysis
 *
 * An ESS is a strategy such that, if adopted by a population,
 * cannot be invaded by any alternative mutant strategy.
 *
 * Mathematical definition:
 * A strategy s* is an ESS if for all s ≠ s*:
 * 1. E(s*, s*) > E(s, s*), or
 * 2. E(s*, s*) = E(s, s*) and E(s*, s) > E(s, s)
 *
 * where E(a, b) is the expected payoff for strategy a against strategy b.
 */

import type { Game, ESSResult } from './types.js';
import { calculatePayoff } from './games.js';

/**
 * ESS Calculator
 */
export class ESSCalculator {
  private game: Game;

  constructor(game: Game) {
    this.game = game;
  }

  /**
   * Check if a pure strategy is an ESS
   */
  isPureESS(strategy: number): boolean {
    const population = new Array(this.game.numStrategies).fill(0);
    population[strategy] = 1;

    const incumbentPayoff = this.game.payoffMatrix[strategy][strategy];

    // Check against all alternative strategies
    for (let mutant = 0; mutant < this.game.numStrategies; mutant++) {
      if (mutant === strategy) continue;

      const mutantPayoff = this.game.payoffMatrix[mutant][strategy];

      // First condition: incumbent does better against itself
      if (incumbentPayoff > mutantPayoff) {
        continue;
      }

      // Second condition: if tied, incumbent does better against mutant
      if (incumbentPayoff === mutantPayoff) {
        const incumbentVsMutant = this.game.payoffMatrix[strategy][mutant];
        const mutantVsMutant = this.game.payoffMatrix[mutant][mutant];

        if (incumbentVsMutant > mutantVsMutant) {
          continue;
        }
      }

      // Failed ESS test
      return false;
    }

    return true;
  }

  /**
   * Check if a mixed strategy is an ESS
   * Uses the stability matrix approach
   */
  isMixedESS(strategy: number[], epsilon: number = 1e-6): ESSResult {
    // Normalize strategy
    const sum = strategy.reduce((a, b) => a + b, 0);
    const normalized = strategy.map((x) => x / sum);

    // Calculate Jacobian matrix at this point
    const jacobian = this.calculateJacobian(normalized);

    // Calculate eigenvalues
    const eigenvalues = this.calculateEigenvalues(jacobian);

    // Check stability: all eigenvalues should have negative real parts
    const isStable = eigenvalues.every((lambda) => lambda < epsilon);

    // Calculate stability margin (most positive eigenvalue)
    const stabilityMargin = Math.max(...eigenvalues);

    return {
      strategy: normalized,
      isStable,
      eigenvalues,
      stabilityMargin,
    };
  }

  /**
   * Calculate Jacobian matrix for replicator dynamics
   */
  private calculateJacobian(population: number[]): number[][] {
    const n = this.game.numStrategies;
    const jacobian: number[][] = Array(n)
      .fill(null)
      .map(() => Array(n).fill(0));

    // Calculate fitness values
    const fitness = population.map((_, i) =>
      calculatePayoff(this.game, i, population)
    );

    // Calculate average fitness
    const avgFitness = population.reduce((sum, x_i, i) =>
      sum + x_i * fitness[i], 0
    );

    // Calculate Jacobian elements
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // ∂(dx_i/dt)/∂x_j
        const payoffDerivative = this.game.payoffMatrix[i][j];
        const avgFitnessDerivative = fitness[j] -
          population[j] * this.game.payoffMatrix[j][j];

        jacobian[i][j] = population[i] * (
          (i === j ? fitness[i] - avgFitness : 0) +
          payoffDerivative - avgFitnessDerivative
        );
      }
    }

    return jacobian;
  }

  /**
   * Calculate eigenvalues of a matrix using QR algorithm
   * (Simplified implementation for 2x2 and 3x3 matrices)
   */
  private calculateEigenvalues(matrix: number[][]): number[] {
    const n = matrix.length;

    if (n === 2) {
      return this.eigenvalues2x2(matrix);
    } else if (n === 3) {
      return this.eigenvalues3x3(matrix);
    } else {
      // For larger matrices, use power iteration for dominant eigenvalue
      return [this.powerIteration(matrix)];
    }
  }

  /**
   * Eigenvalues for 2x2 matrix
   */
  private eigenvalues2x2(matrix: number[][]): number[] {
    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[1][0];
    const d = matrix[1][1];

    const trace = a + d;
    const det = a * d - b * c;

    const discriminant = trace * trace - 4 * det;

    if (discriminant >= 0) {
      const sqrt = Math.sqrt(discriminant);
      return [
        (trace + sqrt) / 2,
        (trace - sqrt) / 2,
      ];
    } else {
      // Complex eigenvalues
      const real = trace / 2;
      const imag = Math.sqrt(-discriminant) / 2;
      return [
        real, // Return real part for stability check
        real,
      ];
    }
  }

  /**
   * Eigenvalues for 3x3 matrix (simplified)
   */
  private eigenvalues3x3(matrix: number[][]): number[] {
    // Use power iteration for dominant eigenvalue
    const lambda1 = this.powerIteration(matrix);

    // Deflate and find second eigenvalue
    const deflated = this.deflateMatrix(matrix, lambda1);
    const lambda2 = this.powerIteration(deflated);

    // Third eigenvalue from trace
    const trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    const lambda3 = trace - lambda1 - lambda2;

    return [lambda1, lambda2, lambda3];
  }

  /**
   * Power iteration to find dominant eigenvalue
   */
  private powerIteration(
    matrix: number[][],
    maxIter: number = 100,
    tolerance: number = 1e-6
  ): number {
    const n = matrix.length;
    let v = Array(n).fill(1);
    let lambda = 0;

    for (let iter = 0; iter < maxIter; iter++) {
      // Matrix-vector multiplication
      const Av = v.map((_, i) =>
        matrix[i].reduce((sum, aij, j) => sum + aij * v[j], 0)
      );

      // New eigenvalue estimate
      const newLambda = this.dotProduct(v, Av) / this.dotProduct(v, v);

      // Normalize
      const norm = Math.sqrt(this.dotProduct(Av, Av));
      v = Av.map((x) => x / norm);

      // Check convergence
      if (Math.abs(newLambda - lambda) < tolerance) {
        return newLambda;
      }

      lambda = newLambda;
    }

    return lambda;
  }

  /**
   * Deflate matrix (remove dominant eigenvalue)
   */
  private deflateMatrix(matrix: number[][], lambda: number): number[][] {
    const n = matrix.length;
    const deflated: number[][] = Array(n)
      .fill(null)
      .map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        deflated[i][j] = matrix[i][j] - (i === j ? lambda : 0);
      }
    }

    return deflated;
  }

  /**
   * Dot product of two vectors
   */
  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  }

  /**
   * Find all pure strategy ESS
   */
  findPureESS(): number[] {
    const essStrategies: number[] = [];

    for (let i = 0; i < this.game.numStrategies; i++) {
      if (this.isPureESS(i)) {
        essStrategies.push(i);
      }
    }

    return essStrategies;
  }

  /**
   * Find mixed strategy ESS by scanning the simplex
   */
  findMixedESS(resolution: number = 10): ESSResult[] {
    const essResults: ESSResult[] = [];

    // Generate candidate strategies
    const candidates = this.generateSimplexPoints(resolution);

    for (const candidate of candidates) {
      const result = this.isMixedESS(candidate);
      if (result.isStable && result.stabilityMargin < -1e-6) {
        // Check if this is a new ESS
        const isNew = !essResults.some((ess) =>
          ess.strategy.every((x, i) => Math.abs(x - result.strategy[i]) < 1e-3)
        );

        if (isNew) {
          essResults.push(result);
        }
      }
    }

    return essResults;
  }

  /**
   * Generate points on the simplex
   */
  private generateSimplexPoints(resolution: number): number[][] {
    const points: number[][] = [];

    const generateRecursive = (
      remaining: number,
      depth: number,
      current: number[]
    ): void => {
      if (depth === this.game.numStrategies - 1) {
        points.push([...current, remaining]);
        return;
      }

      for (let i = 0; i <= remaining * resolution; i++) {
        const value = i / resolution;
        generateRecursive(remaining - value, depth + 1, [...current, value]);
      }
    };

    generateRecursive(1.0, 0, []);
    return points;
  }

  /**
   * Check stability against specific invader
   */
  canInvade(
    resident: number[],
    invader: number[],
    invaderFreq: number = 0.01
  ): boolean {
    // Population after invasion
    const mixed = resident.map((x, i) =>
      x * (1 - invaderFreq) + invader[i] * invaderFreq
    );

    // Payoffs
    const residentPayoff = resident.reduce((sum, x, i) =>
      sum + x * calculatePayoff(this.game, i, mixed), 0
    );

    const invaderPayoff = invader.reduce((sum, x, i) =>
      sum + x * calculatePayoff(this.game, i, mixed), 0
    );

    // Invader succeeds if it has higher payoff
    return invaderPayoff > residentPayoff;
  }

  /**
   * Calculate invasion fitness
   */
  invasionFitness(resident: number[], invader: number[]): number {
    const residentPayoff = invader.reduce((sum, x, i) =>
      sum + x * calculatePayoff(this.game, i, resident), 0
    );

    const incumbentPayoff = resident.reduce((sum, x, i) =>
      sum + x * calculatePayoff(this.game, i, resident), 0
    );

    return residentPayoff - incumbentPayoff;
  }

  /**
   * Find the basin of attraction for an ESS
   */
  findBasinOfAttraction(
    ess: number[],
    resolution: number = 20,
    threshold: number = 0.1
  ): number[][] {
    const basin: number[][] = [];

    const candidates = this.generateSimplexPoints(resolution);

    for (const candidate of candidates) {
      // Check if this point converges to the ESS
      // (This would require running replicator dynamics, simplified here)
      const distance = this.euclideanDistance(candidate, ess);
      if (distance < threshold) {
        basin.push(candidate);
      }
    }

    return basin;
  }

  /**
   * Euclidean distance between two strategies
   */
  private euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(
      a.reduce((sum, ai, i) => sum + Math.pow(ai - b[i], 2), 0)
    );
  }
}

/**
 * Helper function to find all ESS in a game
 */
export function findAllESS(game: Game): {
  pure: number[];
  mixed: ESSResult[];
} {
  const calculator = new ESSCalculator(game);

  return {
    pure: calculator.findPureESS(),
    mixed: calculator.findMixedESS(5),
  };
}
