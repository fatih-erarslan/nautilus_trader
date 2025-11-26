/**
 * Quantum Annealing Simulation
 *
 * Simulates quantum annealing process for combinatorial optimization.
 * Uses quantum tunneling and thermal fluctuations to escape local minima.
 *
 * Applications: Scheduling, Constraint Satisfaction, TSP, Protein Folding
 */

import { create, all } from 'mathjs';
import type { MathJsStatic } from 'mathjs';

const math = create(all) as MathJsStatic;

export interface AnnealingConfig {
  numQubits: number;
  initialTemperature: number;
  finalTemperature: number;
  annealingTime: number;      // Total annealing time
  numSteps: number;            // Number of discrete time steps
  quantumStrength: number;     // Transverse field strength
  method: 'simulated' | 'quantum-monte-carlo' | 'path-integral';
}

export interface QuboMatrix {
  Q: number[][];  // Quadratic Unconstrained Binary Optimization matrix
}

export interface AnnealingResult {
  solution: number[];
  energy: number;
  successProbability: number;
  annealingPath: AnnealingSnapshot[];
  executionTime: number;
  converged: boolean;
}

export interface AnnealingSnapshot {
  time: number;
  temperature: number;
  transverseField: number;
  state: number[];
  energy: number;
  quantumFluctuations: number;
}

/**
 * Quantum Annealing Simulator
 */
export class QuantumAnnealer {
  private config: AnnealingConfig;
  private qubo: QuboMatrix;
  private currentState: number[];
  private annealingPath: AnnealingSnapshot[] = [];

  constructor(config: AnnealingConfig, qubo: QuboMatrix) {
    this.config = config;
    this.qubo = qubo;
    this.currentState = this.initializeState();
  }

  /**
   * Run quantum annealing optimization
   */
  async anneal(): Promise<AnnealingResult> {
    const startTime = Date.now();

    // Reset state
    this.currentState = this.initializeState();
    this.annealingPath = [];

    let bestSolution = [...this.currentState];
    let bestEnergy = this.computeEnergy(this.currentState);

    const timeStep = this.config.annealingTime / this.config.numSteps;

    for (let step = 0; step < this.config.numSteps; step++) {
      const t = step / this.config.numSteps; // Normalized time [0, 1]

      // Annealing schedule
      const temperature = this.temperatureSchedule(t);
      const transverseField = this.transverseFieldSchedule(t);

      // Perform quantum annealing step based on method
      switch (this.config.method) {
        case 'quantum-monte-carlo':
          this.quantumMonteCarloStep(temperature, transverseField);
          break;
        case 'path-integral':
          this.pathIntegralStep(temperature, transverseField);
          break;
        default:
          this.simulatedAnnealingStep(temperature, transverseField);
      }

      const currentEnergy = this.computeEnergy(this.currentState);

      // Track best solution found
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        bestSolution = [...this.currentState];
      }

      // Record snapshot
      this.annealingPath.push({
        time: step * timeStep,
        temperature,
        transverseField,
        state: [...this.currentState],
        energy: currentEnergy,
        quantumFluctuations: this.computeQuantumFluctuations(transverseField)
      });
    }

    const executionTime = Date.now() - startTime;
    const converged = this.checkConvergence();
    const successProbability = this.estimateSuccessProbability(bestEnergy);

    return {
      solution: bestSolution,
      energy: bestEnergy,
      successProbability,
      annealingPath: this.annealingPath,
      executionTime,
      converged
    };
  }

  /**
   * Initialize state (random or ground state)
   */
  private initializeState(): number[] {
    // Start in uniform superposition (simulated as random state)
    return Array(this.config.numQubits)
      .fill(0)
      .map(() => Math.random() > 0.5 ? 1 : 0);
  }

  /**
   * Temperature schedule: T(t) = T_initial * (1 - t) + T_final * t
   */
  private temperatureSchedule(t: number): number {
    return this.config.initialTemperature * (1 - t) + this.config.finalTemperature * t;
  }

  /**
   * Transverse field schedule: Γ(t) = Γ_0 * (1 - t)
   * Strong at start (quantum tunneling), weak at end (classical)
   */
  private transverseFieldSchedule(t: number): number {
    return this.config.quantumStrength * (1 - t);
  }

  /**
   * Simulated annealing step with quantum tunneling
   */
  private simulatedAnnealingStep(temperature: number, transverseField: number): void {
    const n = this.config.numQubits;

    // Try flipping each qubit
    for (let i = 0; i < n; i++) {
      const newState = [...this.currentState];
      newState[i] = 1 - newState[i]; // Flip bit

      const currentEnergy = this.computeEnergy(this.currentState);
      const newEnergy = this.computeEnergy(newState);
      const deltaE = newEnergy - currentEnergy;

      // Acceptance probability with quantum tunneling term
      const classicalProb = Math.exp(-deltaE / temperature);
      const quantumProb = transverseField / (transverseField + Math.abs(deltaE));
      const acceptProb = Math.max(classicalProb, quantumProb);

      if (deltaE < 0 || Math.random() < acceptProb) {
        this.currentState = newState;
      }
    }
  }

  /**
   * Quantum Monte Carlo step using path integral formulation
   */
  private quantumMonteCarloStep(temperature: number, transverseField: number): void {
    const numTrotterSlices = 10; // Discretize imaginary time

    // Perform updates on each Trotter slice
    for (let slice = 0; slice < numTrotterSlices; slice++) {
      this.trotterSliceUpdate(temperature, transverseField, slice, numTrotterSlices);
    }
  }

  /**
   * Update single Trotter slice in QMC
   */
  private trotterSliceUpdate(
    temperature: number,
    transverseField: number,
    slice: number,
    numSlices: number
  ): void {
    const n = this.config.numQubits;

    for (let i = 0; i < n; i++) {
      const newState = [...this.currentState];
      newState[i] = 1 - newState[i];

      const classicalEnergy = this.computeEnergy(newState) - this.computeEnergy(this.currentState);

      // Coupling to neighboring time slices (quantum tunneling)
      const kineticEnergy = -transverseField * Math.log(Math.tanh(1 / (temperature * numSlices)));

      const totalDeltaE = classicalEnergy + kineticEnergy;
      const acceptProb = Math.min(1, Math.exp(-totalDeltaE / temperature));

      if (Math.random() < acceptProb) {
        this.currentState = newState;
      }
    }
  }

  /**
   * Path integral formulation step
   */
  private pathIntegralStep(temperature: number, transverseField: number): void {
    // Simplified path integral using effective action
    const n = this.config.numQubits;

    for (let i = 0; i < n; i++) {
      const newState = [...this.currentState];
      newState[i] = 1 - newState[i];

      const classicalAction = this.computeEnergy(newState) - this.computeEnergy(this.currentState);
      const quantumAction = -transverseField * (2 * newState[i] - 1);
      const totalAction = classicalAction + quantumAction;

      const acceptProb = Math.min(1, Math.exp(-totalAction / temperature));

      if (Math.random() < acceptProb) {
        this.currentState = newState;
      }
    }
  }

  /**
   * Compute energy E = x^T Q x for QUBO problem
   */
  private computeEnergy(state: number[]): number {
    let energy = 0;
    const n = state.length;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        energy += state[i] * this.qubo.Q[i][j] * state[j];
      }
    }

    return energy;
  }

  /**
   * Compute quantum fluctuations strength
   */
  private computeQuantumFluctuations(transverseField: number): number {
    // Estimate quantum fluctuations as √⟨(ΔΓ)²⟩
    return Math.sqrt(transverseField * this.config.numQubits);
  }

  /**
   * Check if annealing converged to ground state
   */
  private checkConvergence(): boolean {
    if (this.annealingPath.length < 10) return false;

    // Check if energy stabilized in final 10% of steps
    const finalSteps = Math.floor(this.annealingPath.length * 0.1);
    const recentEnergies = this.annealingPath
      .slice(-finalSteps)
      .map(snapshot => snapshot.energy);

    const energyVariance = this.variance(recentEnergies);
    return energyVariance < 1e-6;
  }

  /**
   * Estimate success probability (reaching ground state)
   */
  private estimateSuccessProbability(energy: number): number {
    // Use Boltzmann distribution at final temperature
    const temperature = this.config.finalTemperature;

    // Estimate ground state energy (lower bound)
    const minPossibleEnergy = -this.config.numQubits * 10; // Rough estimate

    // Probability proportional to exp(-E/T)
    const prob = Math.exp(-(energy - minPossibleEnergy) / Math.max(temperature, 0.01));

    return Math.min(1, Math.max(0, prob));
  }

  private variance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }
}

/**
 * Create QUBO problem from various formulations
 */
export class QUBOFormulator {
  /**
   * Convert MaxCut to QUBO
   */
  static maxCutToQUBO(edges: [number, number, number][]): QuboMatrix {
    const numNodes = Math.max(...edges.flatMap(([i, j]) => [i, j])) + 1;
    const Q = Array(numNodes).fill(0).map(() => Array(numNodes).fill(0));

    edges.forEach(([i, j, weight]) => {
      // QUBO formulation: minimize -∑w_ij(1 - 2x_i)(1 - 2x_j)
      Q[i][i] -= weight;
      Q[j][j] -= weight;
      Q[i][j] += 2 * weight;
      Q[j][i] += 2 * weight;
    });

    return { Q };
  }

  /**
   * Convert TSP to QUBO
   */
  static tspToQUBO(distanceMatrix: number[][]): QuboMatrix {
    const n = distanceMatrix.length;
    const numVars = n * n; // x_ij: city j visited at position i
    const Q = Array(numVars).fill(0).map(() => Array(numVars).fill(0));

    const penalty = 1000; // Large penalty for constraint violations

    // Objective: minimize total distance
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          if (k < n - 1) {
            const idx1 = i * n + j;
            const idx2 = (i + 1) * n + k;
            Q[idx1][idx2] += distanceMatrix[j][k];
          }
        }
      }
    }

    // Constraint: each city visited exactly once
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const idx = i * n + j;
        Q[idx][idx] += penalty * (1 - 2 * n);
        for (let i2 = i + 1; i2 < n; i2++) {
          const idx2 = i2 * n + j;
          Q[idx][idx2] += 2 * penalty;
        }
      }
    }

    // Constraint: each position has exactly one city
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const idx = i * n + j;
        Q[idx][idx] += penalty * (1 - 2 * n);
        for (let j2 = j + 1; j2 < n; j2++) {
          const idx2 = i * n + j2;
          Q[idx][idx2] += 2 * penalty;
        }
      }
    }

    return { Q };
  }

  /**
   * Convert constraint satisfaction to QUBO
   */
  static constraintSatisfactionToQUBO(
    numVars: number,
    constraints: Array<{ vars: number[]; coeffs: number[]; rhs: number }>
  ): QuboMatrix {
    const Q = Array(numVars).fill(0).map(() => Array(numVars).fill(0));

    // Each constraint: (∑c_i x_i - rhs)² = 0
    constraints.forEach(({ vars, coeffs, rhs }) => {
      // Expand (∑c_i x_i - rhs)²
      for (let i = 0; i < vars.length; i++) {
        for (let j = 0; j < vars.length; j++) {
          Q[vars[i]][vars[j]] += coeffs[i] * coeffs[j];
        }
        Q[vars[i]][vars[i]] -= 2 * coeffs[i] * rhs;
      }
      // Constant term (rhs²) doesn't affect optimization
    });

    return { Q };
  }

  /**
   * Convert portfolio optimization to QUBO
   */
  static portfolioToQUBO(
    returns: number[],
    covarianceMatrix: number[][],
    budget: number,
    riskAversion: number
  ): QuboMatrix {
    const n = returns.length;
    const Q = Array(n).fill(0).map(() => Array(n).fill(0));

    // Objective: maximize return - λ * risk
    // = maximize ∑r_i x_i - λ * ∑∑σ_ij x_i x_j
    // Convert to minimization: -∑r_i x_i + λ * ∑∑σ_ij x_i x_j

    for (let i = 0; i < n; i++) {
      Q[i][i] -= returns[i]; // Linear return term
      Q[i][i] += riskAversion * covarianceMatrix[i][i]; // Variance term

      for (let j = i + 1; j < n; j++) {
        Q[i][j] += riskAversion * covarianceMatrix[i][j]; // Covariance term
        Q[j][i] += riskAversion * covarianceMatrix[j][i];
      }
    }

    // Budget constraint: ∑x_i = budget
    const penalty = 100;
    for (let i = 0; i < n; i++) {
      Q[i][i] += penalty * (1 - 2 * budget);
      for (let j = i + 1; j < n; j++) {
        Q[i][j] += 2 * penalty;
        Q[j][i] += 2 * penalty;
      }
    }

    return { Q };
  }
}

/**
 * Solve QUBO problem using quantum annealing
 */
export async function solveQUBO(
  qubo: QuboMatrix,
  config: Partial<AnnealingConfig> = {}
): Promise<AnnealingResult> {
  const numQubits = qubo.Q.length;

  const fullConfig: AnnealingConfig = {
    numQubits,
    initialTemperature: config.initialTemperature || 100,
    finalTemperature: config.finalTemperature || 0.01,
    annealingTime: config.annealingTime || 1000,
    numSteps: config.numSteps || 1000,
    quantumStrength: config.quantumStrength || 10,
    method: config.method || 'quantum-monte-carlo'
  };

  const annealer = new QuantumAnnealer(fullConfig, qubo);
  return annealer.anneal();
}

/**
 * Solve MaxCut using quantum annealing
 */
export async function solveMaxCutAnnealing(
  edges: [number, number, number][],
  config: Partial<AnnealingConfig> = {}
): Promise<AnnealingResult> {
  const qubo = QUBOFormulator.maxCutToQUBO(edges);
  return solveQUBO(qubo, config);
}

/**
 * Solve TSP using quantum annealing
 */
export async function solveTSPAnnealing(
  distanceMatrix: number[][],
  config: Partial<AnnealingConfig> = {}
): Promise<AnnealingResult> {
  const qubo = QUBOFormulator.tspToQUBO(distanceMatrix);
  return solveQUBO(qubo, config);
}
