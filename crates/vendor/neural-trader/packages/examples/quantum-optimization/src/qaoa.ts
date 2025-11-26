/**
 * Quantum Approximate Optimization Algorithm (QAOA)
 *
 * Hybrid quantum-classical algorithm for combinatorial optimization problems.
 * Uses parameterized quantum circuits with classical optimization of angles.
 *
 * Applications: MaxCut, Graph Coloring, TSP, Portfolio Optimization
 */

import { create, all } from 'mathjs';
import type { MathJsStatic } from 'mathjs';

const math = create(all) as MathJsStatic;

export interface QAOAConfig {
  numQubits: number;
  depth: number;           // Number of QAOA layers (p)
  maxIterations: number;   // Classical optimization iterations
  learningRate: number;
  tolerance: number;
}

export interface QAOAProblem {
  type: 'maxcut' | 'tsp' | 'portfolio' | 'constraint-satisfaction';
  costMatrix: number[][];  // Problem encoding
  constraints?: any[];     // Optional constraints
}

export interface QAOAResult {
  bestSolution: number[];
  bestEnergy: number;
  optimalAngles: { beta: number[]; gamma: number[] };
  iterations: number;
  converged: boolean;
  stateVector?: Complex[];
  executionTime: number;
}

export interface Complex {
  re: number;
  im: number;
}

/**
 * QAOA Optimizer using simulated quantum circuits
 */
export class QAOAOptimizer {
  private config: QAOAConfig;
  private problem: QAOAProblem;

  constructor(config: QAOAConfig, problem: QAOAProblem) {
    this.config = config;
    this.problem = problem;
  }

  /**
   * Run QAOA optimization
   */
  async optimize(): Promise<QAOAResult> {
    const startTime = Date.now();

    // Initialize parameters (beta and gamma angles)
    let beta = Array(this.config.depth).fill(0).map(() => Math.random() * Math.PI);
    let gamma = Array(this.config.depth).fill(0).map(() => Math.random() * 2 * Math.PI);

    let bestEnergy = Infinity;
    let bestSolution: number[] = [];
    let converged = false;

    for (let iter = 0; iter < this.config.maxIterations; iter++) {
      // Apply QAOA circuit and measure expectation value
      const { energy, solution, stateVector } = this.evaluateCircuit(beta, gamma);

      if (energy < bestEnergy) {
        bestEnergy = energy;
        bestSolution = solution;
      }

      // Check convergence
      if (iter > 0 && Math.abs(energy - bestEnergy) < this.config.tolerance) {
        converged = true;
        break;
      }

      // Classical optimization of parameters (gradient descent)
      const gradients = this.computeGradients(beta, gamma);
      beta = beta.map((b, i) => b - this.config.learningRate * gradients.beta[i]);
      gamma = gamma.map((g, i) => g - this.config.learningRate * gradients.gamma[i]);

      // Keep angles in valid range
      beta = beta.map(b => ((b % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
      gamma = gamma.map(g => ((g % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
    }

    const executionTime = Date.now() - startTime;

    return {
      bestSolution,
      bestEnergy,
      optimalAngles: { beta, gamma },
      iterations: this.config.maxIterations,
      converged,
      executionTime
    };
  }

  /**
   * Evaluate QAOA circuit for given parameters
   */
  private evaluateCircuit(beta: number[], gamma: number[]): {
    energy: number;
    solution: number[];
    stateVector: Complex[];
  } {
    // Initialize state vector to uniform superposition |+⟩^n
    let stateVector = this.initializeState();

    // Apply QAOA layers
    for (let p = 0; p < this.config.depth; p++) {
      // Apply problem Hamiltonian (cost function)
      stateVector = this.applyCostHamiltonian(stateVector, gamma[p]);

      // Apply mixer Hamiltonian (X rotations)
      stateVector = this.applyMixerHamiltonian(stateVector, beta[p]);
    }

    // Measure expectation value of cost function
    const energy = this.computeExpectation(stateVector);

    // Sample most probable bitstring
    const solution = this.sampleSolution(stateVector);

    return { energy, solution, stateVector };
  }

  /**
   * Initialize quantum state to uniform superposition
   */
  private initializeState(): Complex[] {
    const n = this.config.numQubits;
    const dim = Math.pow(2, n);
    const amplitude = 1 / Math.sqrt(dim);

    return Array(dim).fill(0).map(() => ({ re: amplitude, im: 0 }));
  }

  /**
   * Apply cost Hamiltonian (problem-specific)
   */
  private applyCostHamiltonian(state: Complex[], gamma: number): Complex[] {
    const newState = [...state];
    const n = this.config.numQubits;

    // For MaxCut: apply Z_i Z_j rotations for each edge
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const weight = this.problem.costMatrix[i]?.[j] || 0;
        if (weight !== 0) {
          newState.forEach((_, idx) => {
            const bit_i = (idx >> i) & 1;
            const bit_j = (idx >> j) & 1;
            const phase = (bit_i === bit_j) ? -1 : 1;
            const angle = gamma * weight * phase;

            newState[idx] = this.complexMult(newState[idx], {
              re: Math.cos(angle),
              im: -Math.sin(angle)
            });
          });
        }
      }
    }

    return newState;
  }

  /**
   * Apply mixer Hamiltonian (X rotations)
   */
  private applyMixerHamiltonian(state: Complex[], beta: number): Complex[] {
    const n = this.config.numQubits;
    let newState = [...state];

    // Apply X rotation to each qubit
    for (let qubit = 0; qubit < n; qubit++) {
      newState = this.applyXRotation(newState, qubit, beta);
    }

    return newState;
  }

  /**
   * Apply X rotation to a single qubit
   */
  private applyXRotation(state: Complex[], qubit: number, angle: number): Complex[] {
    const dim = state.length;
    const newState = Array(dim).fill({ re: 0, im: 0 });
    const cos = Math.cos(angle / 2);
    const sin = Math.sin(angle / 2);

    for (let idx = 0; idx < dim; idx++) {
      const flippedIdx = idx ^ (1 << qubit); // Flip bit at position 'qubit'

      // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
      const term1 = this.complexScale(state[idx], cos);
      const term2 = this.complexMult(state[flippedIdx], { re: 0, im: -sin });

      newState[idx] = this.complexAdd(
        newState[idx],
        this.complexAdd(term1, term2)
      );
    }

    return newState;
  }

  /**
   * Compute expectation value of cost function
   */
  private computeExpectation(state: Complex[]): number {
    let expectation = 0;
    const n = this.config.numQubits;

    state.forEach((amplitude, idx) => {
      const probability = amplitude.re ** 2 + amplitude.im ** 2;
      const bitstring = this.indexToBitstring(idx, n);
      const cost = this.evaluateCost(bitstring);
      expectation += probability * cost;
    });

    return expectation;
  }

  /**
   * Sample most probable solution from state vector
   */
  private sampleSolution(state: Complex[]): number[] {
    let maxProb = 0;
    let maxIdx = 0;

    state.forEach((amplitude, idx) => {
      const prob = amplitude.re ** 2 + amplitude.im ** 2;
      if (prob > maxProb) {
        maxProb = prob;
        maxIdx = idx;
      }
    });

    return this.indexToBitstring(maxIdx, this.config.numQubits);
  }

  /**
   * Evaluate cost function for a bitstring
   */
  private evaluateCost(bitstring: number[]): number {
    const n = this.config.numQubits;
    let cost = 0;

    // MaxCut cost: sum of edge weights where endpoints have different values
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const weight = this.problem.costMatrix[i]?.[j] || 0;
        if (bitstring[i] !== bitstring[j]) {
          cost -= weight; // Negative because we want to maximize cut
        }
      }
    }

    return cost;
  }

  /**
   * Compute gradients using parameter shift rule
   */
  private computeGradients(beta: number[], gamma: number[]): {
    beta: number[];
    gamma: number[];
  } {
    const shift = Math.PI / 2;
    const betaGrad: number[] = [];
    const gammaGrad: number[] = [];

    // Gradient for beta parameters
    for (let i = 0; i < beta.length; i++) {
      const betaPlus = [...beta];
      const betaMinus = [...beta];
      betaPlus[i] += shift;
      betaMinus[i] -= shift;

      const energyPlus = this.evaluateCircuit(betaPlus, gamma).energy;
      const energyMinus = this.evaluateCircuit(betaMinus, gamma).energy;

      betaGrad.push((energyPlus - energyMinus) / 2);
    }

    // Gradient for gamma parameters
    for (let i = 0; i < gamma.length; i++) {
      const gammaPlus = [...gamma];
      const gammaMinus = [...gamma];
      gammaPlus[i] += shift;
      gammaMinus[i] -= shift;

      const energyPlus = this.evaluateCircuit(beta, gammaPlus).energy;
      const energyMinus = this.evaluateCircuit(beta, gammaMinus).energy;

      gammaGrad.push((energyPlus - energyMinus) / 2);
    }

    return { beta: betaGrad, gamma: gammaGrad };
  }

  // Complex number operations
  private complexAdd(a: Complex, b: Complex): Complex {
    return { re: a.re + b.re, im: a.im + b.im };
  }

  private complexMult(a: Complex, b: Complex): Complex {
    return {
      re: a.re * b.re - a.im * b.im,
      im: a.re * b.im + a.im * b.re
    };
  }

  private complexScale(a: Complex, scalar: number): Complex {
    return { re: a.re * scalar, im: a.im * scalar };
  }

  private indexToBitstring(idx: number, n: number): number[] {
    return Array(n).fill(0).map((_, i) => (idx >> i) & 1);
  }
}

/**
 * Helper function to create MaxCut problem
 */
export function createMaxCutProblem(edges: [number, number, number][]): QAOAProblem {
  const numQubits = Math.max(...edges.flatMap(([i, j]) => [i, j])) + 1;
  const costMatrix = Array(numQubits).fill(0).map(() => Array(numQubits).fill(0));

  edges.forEach(([i, j, weight]) => {
    costMatrix[i][j] = weight;
    costMatrix[j][i] = weight;
  });

  return {
    type: 'maxcut',
    costMatrix
  };
}

/**
 * Solve MaxCut problem using QAOA
 */
export async function solveMaxCut(
  edges: [number, number, number][],
  config: Partial<QAOAConfig> = {}
): Promise<QAOAResult> {
  const problem = createMaxCutProblem(edges);
  const numQubits = problem.costMatrix.length;

  const fullConfig: QAOAConfig = {
    numQubits,
    depth: config.depth || 3,
    maxIterations: config.maxIterations || 100,
    learningRate: config.learningRate || 0.1,
    tolerance: config.tolerance || 1e-6
  };

  const optimizer = new QAOAOptimizer(fullConfig, problem);
  return optimizer.optimize();
}
