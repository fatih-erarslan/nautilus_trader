/**
 * @neural-trader/example-quantum-optimization
 *
 * Quantum-inspired optimization algorithms with swarm-based circuit exploration
 * for combinatorial and constraint problems.
 *
 * Features:
 * - QAOA (Quantum Approximate Optimization Algorithm)
 * - VQE (Variational Quantum Eigensolver)
 * - Quantum Annealing simulation
 * - Swarm-based circuit exploration with AgentDB
 * - Memory-based pattern learning
 * - OpenRouter integration for problem decomposition
 *
 * Applications:
 * - MaxCut and graph optimization
 * - Traveling Salesman Problem (TSP)
 * - Portfolio optimization
 * - Constraint satisfaction problems
 * - Molecular simulation (VQE)
 * - Scheduling and logistics
 */

// QAOA exports
export {
  QAOAOptimizer,
  type QAOAConfig,
  type QAOAProblem,
  type QAOAResult,
  createMaxCutProblem,
  solveMaxCut
} from './qaoa.js';

// VQE exports
export {
  VQESolver,
  type VQEConfig,
  type Hamiltonian,
  type PauliString,
  type VQEResult,
  createIsingHamiltonian
} from './vqe.js';

// Quantum Annealing exports
export {
  QuantumAnnealer,
  QUBOFormulator,
  type AnnealingConfig,
  type QuboMatrix,
  type AnnealingResult,
  type AnnealingSnapshot,
  solveQUBO,
  solveMaxCutAnnealing,
  solveTSPAnnealing
} from './quantum-annealing.js';

// Swarm Circuit Exploration exports
export {
  SwarmCircuitExplorer,
  type CircuitExplorationConfig,
  type QuantumCircuit,
  type Gate,
  type CircuitMetadata,
  type SwarmAgent,
  type ExplorationResult,
  type CircuitPattern,
  exploreCircuits
} from './swarm-circuits.js';

// Common types
export interface Complex {
  re: number;
  im: number;
}

/**
 * Unified quantum optimization interface
 */
export class QuantumOptimizer {
  /**
   * Solve MaxCut problem using best available method
   */
  static async solveMaxCut(
    edges: [number, number, number][],
    method: 'qaoa' | 'annealing' | 'auto' = 'auto'
  ): Promise<{
    solution: number[];
    energy: number;
    method: string;
    executionTime: number;
  }> {
    const numNodes = Math.max(...edges.flatMap(([i, j]) => [i, j])) + 1;

    // Auto-select method based on problem size
    if (method === 'auto') {
      method = numNodes <= 10 ? 'qaoa' : 'annealing';
    }

    if (method === 'qaoa') {
      const { solveMaxCut } = await import('./qaoa.js');
      const result = await solveMaxCut(edges, { depth: 3, maxIterations: 100 });

      return {
        solution: result.bestSolution,
        energy: result.bestEnergy,
        method: 'QAOA',
        executionTime: result.executionTime
      };
    } else {
      const { solveMaxCutAnnealing } = await import('./quantum-annealing.js');
      const result = await solveMaxCutAnnealing(edges, { numSteps: 1000 });

      return {
        solution: result.solution,
        energy: result.energy,
        method: 'Quantum Annealing',
        executionTime: result.executionTime
      };
    }
  }

  /**
   * Solve TSP using quantum annealing
   */
  static async solveTSP(
    distanceMatrix: number[][]
  ): Promise<{
    tour: number[];
    distance: number;
    executionTime: number;
  }> {
    const { solveTSPAnnealing } = await import('./quantum-annealing.js');
    const result = await solveTSPAnnealing(distanceMatrix, {
      numSteps: 2000,
      method: 'quantum-monte-carlo'
    });

    return {
      tour: result.solution,
      distance: result.energy,
      executionTime: result.executionTime
    };
  }

  /**
   * Find ground state energy using VQE
   */
  static async findGroundState(
    hamiltonian: import('./vqe.js').Hamiltonian,
    ansatzType: 'hardware-efficient' | 'uccsd' = 'hardware-efficient'
  ): Promise<{
    energy: number;
    state: Complex[];
    parameters: number[];
    executionTime: number;
  }> {
    const { VQESolver } = await import('./vqe.js');

    const numQubits = hamiltonian.pauliStrings[0]?.pauli.length || 4;

    const solver = new VQESolver(
      {
        numQubits,
        ansatzType,
        ansatzDepth: 3,
        maxIterations: 100,
        optimizer: 'adam',
        learningRate: 0.1,
        tolerance: 1e-6
      },
      hamiltonian
    );

    const result = await solver.solve();

    return {
      energy: result.groundStateEnergy,
      state: result.groundState,
      parameters: result.optimalParameters,
      executionTime: result.executionTime
    };
  }

  /**
   * Explore quantum circuits for optimization
   */
  static async exploreCircuits(config: {
    numQubits: number;
    problemType: 'maxcut' | 'vqe' | 'qaoa';
    swarmSize?: number;
    explorationSteps?: number;
  }): Promise<{
    bestCircuit: import('./swarm-circuits.js').QuantumCircuit;
    performance: number;
    learnedPatterns: import('./swarm-circuits.js').CircuitPattern[];
    executionTime: number;
  }> {
    const { exploreCircuits } = await import('./swarm-circuits.js');

    const result = await exploreCircuits({
      numQubits: config.numQubits,
      problemType: config.problemType,
      swarmSize: config.swarmSize || 20,
      explorationSteps: config.explorationSteps || 100,
      maxDepth: 5,
      learningRate: 0.01,
      memorySize: 1000
    });

    return {
      bestCircuit: result.bestCircuit,
      performance: result.bestPerformance,
      learnedPatterns: result.learnedPatterns,
      executionTime: result.executionTime
    };
  }

  /**
   * Optimize portfolio using quantum-inspired methods
   */
  static async optimizePortfolio(
    returns: number[],
    covarianceMatrix: number[][],
    budget: number,
    riskAversion: number = 1.0
  ): Promise<{
    allocation: number[];
    expectedReturn: number;
    risk: number;
    executionTime: number;
  }> {
    const { QUBOFormulator, solveQUBO } = await import('./quantum-annealing.js');

    const qubo = QUBOFormulator.portfolioToQUBO(
      returns,
      covarianceMatrix,
      budget,
      riskAversion
    );

    const result = await solveQUBO(qubo, {
      numSteps: 1500,
      method: 'quantum-monte-carlo'
    });

    // Compute portfolio metrics
    const allocation = result.solution;
    const expectedReturn = returns.reduce((sum, r, i) => sum + r * allocation[i], 0);

    let risk = 0;
    for (let i = 0; i < allocation.length; i++) {
      for (let j = 0; j < allocation.length; j++) {
        risk += allocation[i] * covarianceMatrix[i][j] * allocation[j];
      }
    }
    risk = Math.sqrt(risk);

    return {
      allocation,
      expectedReturn,
      risk,
      executionTime: result.executionTime
    };
  }

  /**
   * Solve constraint satisfaction problem
   */
  static async solveConstraintSatisfaction(
    numVars: number,
    constraints: Array<{ vars: number[]; coeffs: number[]; rhs: number }>
  ): Promise<{
    solution: number[];
    satisfied: boolean;
    executionTime: number;
  }> {
    const { QUBOFormulator, solveQUBO } = await import('./quantum-annealing.js');

    const qubo = QUBOFormulator.constraintSatisfactionToQUBO(numVars, constraints);
    const result = await solveQUBO(qubo, { numSteps: 1000 });

    // Check if constraints are satisfied
    const solution = result.solution;
    const satisfied = constraints.every(({ vars, coeffs, rhs }) => {
      const sum = vars.reduce((s, v, i) => s + coeffs[i] * solution[v], 0);
      return Math.abs(sum - rhs) < 0.1;
    });

    return {
      solution,
      satisfied,
      executionTime: result.executionTime
    };
  }
}

/**
 * Compare quantum vs classical optimization
 */
export class QuantumClassicalComparison {
  /**
   * Compare QAOA vs classical MaxCut solver
   */
  static async compareMaxCut(
    edges: [number, number, number][]
  ): Promise<{
    quantum: { solution: number[]; energy: number; time: number };
    classical: { solution: number[]; energy: number; time: number };
    speedup: number;
    qualityRatio: number;
  }> {
    // Quantum solution
    const quantumStart = Date.now();
    const { solveMaxCut } = await import('./qaoa.js');
    const quantumResult = await solveMaxCut(edges, { depth: 3, maxIterations: 50 });
    const quantumTime = Date.now() - quantumStart;

    // Classical solution (greedy)
    const classicalStart = Date.now();
    const classicalResult = this.greedyMaxCut(edges);
    const classicalTime = Date.now() - classicalStart;

    const speedup = classicalTime / quantumTime;
    const qualityRatio = Math.abs(quantumResult.bestEnergy) / Math.abs(classicalResult.energy);

    return {
      quantum: {
        solution: quantumResult.bestSolution,
        energy: quantumResult.bestEnergy,
        time: quantumTime
      },
      classical: {
        solution: classicalResult.solution,
        energy: classicalResult.energy,
        time: classicalTime
      },
      speedup,
      qualityRatio
    };
  }

  /**
   * Classical greedy MaxCut solver
   */
  private static greedyMaxCut(
    edges: [number, number, number][]
  ): { solution: number[]; energy: number } {
    const numNodes = Math.max(...edges.flatMap(([i, j]) => [i, j])) + 1;
    const solution = Array(numNodes).fill(0);

    // Greedy assignment
    for (let node = 0; node < numNodes; node++) {
      // Try both assignments and pick better one
      solution[node] = 0;
      const energy0 = this.evaluateMaxCut(edges, solution);

      solution[node] = 1;
      const energy1 = this.evaluateMaxCut(edges, solution);

      solution[node] = energy1 < energy0 ? 1 : 0;
    }

    const energy = this.evaluateMaxCut(edges, solution);

    return { solution, energy };
  }

  /**
   * Evaluate MaxCut solution
   */
  private static evaluateMaxCut(
    edges: [number, number, number][],
    solution: number[]
  ): number {
    let energy = 0;

    edges.forEach(([i, j, weight]) => {
      if (solution[i] !== solution[j]) {
        energy -= weight; // Negative because we want to maximize
      }
    });

    return energy;
  }
}

// Example usage demonstrations
export const examples = {
  /**
   * Example: Solve MaxCut problem
   */
  async maxcut() {
    const edges: [number, number, number][] = [
      [0, 1, 1],
      [0, 2, 1],
      [1, 2, 1],
      [1, 3, 1],
      [2, 3, 1]
    ];

    const result = await QuantumOptimizer.solveMaxCut(edges, 'qaoa');
    console.log('MaxCut Solution:', result);

    return result;
  },

  /**
   * Example: Explore quantum circuits
   */
  async circuitExploration() {
    const result = await QuantumOptimizer.exploreCircuits({
      numQubits: 4,
      problemType: 'maxcut',
      swarmSize: 15,
      explorationSteps: 50
    });

    console.log('Best Circuit Depth:', result.bestCircuit.depth);
    console.log('Performance:', result.performance);
    console.log('Learned Patterns:', result.learnedPatterns.length);

    return result;
  },

  /**
   * Example: Portfolio optimization
   */
  async portfolio() {
    const returns = [0.12, 0.10, 0.08, 0.15];
    const covariance = [
      [0.04, 0.02, 0.01, 0.03],
      [0.02, 0.03, 0.01, 0.02],
      [0.01, 0.01, 0.02, 0.01],
      [0.03, 0.02, 0.01, 0.05]
    ];

    const result = await QuantumOptimizer.optimizePortfolio(
      returns,
      covariance,
      2, // Budget: invest in 2 assets
      1.5 // Risk aversion
    );

    console.log('Optimal Allocation:', result.allocation);
    console.log('Expected Return:', result.expectedReturn);
    console.log('Risk (StdDev):', result.risk);

    return result;
  },

  /**
   * Example: Compare quantum vs classical
   */
  async comparison() {
    const edges: [number, number, number][] = [
      [0, 1, 1],
      [0, 2, 1],
      [1, 2, 1],
      [1, 3, 1],
      [2, 3, 1],
      [0, 3, 1]
    ];

    const comparison = await QuantumClassicalComparison.compareMaxCut(edges);

    console.log('Quantum Energy:', comparison.quantum.energy);
    console.log('Classical Energy:', comparison.classical.energy);
    console.log('Speedup:', comparison.speedup.toFixed(2) + 'x');
    console.log('Quality Ratio:', comparison.qualityRatio.toFixed(2));

    return comparison;
  }
};

// Default export
export default QuantumOptimizer;
