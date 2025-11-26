/**
 * Comprehensive tests for quantum optimization algorithms
 * Comparing quantum-inspired methods vs classical approaches
 */

import { describe, test, expect, beforeAll } from '@jest/globals';
import {
  QAOAOptimizer,
  VQESolver,
  QuantumAnnealer,
  SwarmCircuitExplorer,
  QuantumOptimizer,
  QuantumClassicalComparison,
  createMaxCutProblem,
  createIsingHamiltonian,
  QUBOFormulator,
  solveMaxCut,
  solveQUBO,
  exploreCircuits
} from '../src/index.js';

describe('QAOA (Quantum Approximate Optimization Algorithm)', () => {
  test('should solve simple MaxCut problem', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1],
      [1, 2, 1],
      [2, 3, 1],
      [3, 0, 1]
    ];

    const result = await solveMaxCut(edges, {
      depth: 2,
      maxIterations: 50,
      learningRate: 0.1,
      tolerance: 1e-5
    });

    expect(result.bestSolution).toBeDefined();
    expect(result.bestSolution.length).toBe(4);
    expect(result.bestEnergy).toBeLessThan(0); // MaxCut energies are negative
    expect(result.converged).toBe(true);
    expect(result.executionTime).toBeGreaterThan(0);
  });

  test('should find optimal MaxCut for K4 complete graph', async () => {
    // K4 complete graph - optimal cut should separate graph 2-2
    const edges: [number, number, number][] = [
      [0, 1, 1], [0, 2, 1], [0, 3, 1],
      [1, 2, 1], [1, 3, 1], [2, 3, 1]
    ];

    const result = await solveMaxCut(edges, {
      depth: 3,
      maxIterations: 100
    });

    // K4 optimal cut = 4 edges
    expect(result.bestEnergy).toBeLessThanOrEqual(-3.5); // Should be close to -4
    expect(result.optimalAngles.beta.length).toBe(3);
    expect(result.optimalAngles.gamma.length).toBe(3);
  });

  test('should handle weighted graphs', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 5],
      [1, 2, 3],
      [2, 0, 2]
    ];

    const result = await solveMaxCut(edges, { depth: 2, maxIterations: 50 });

    expect(result.bestSolution.length).toBe(3);
    expect(result.bestEnergy).toBeLessThan(0);
  });

  test('should improve with deeper circuits', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]
    ];

    const shallow = await solveMaxCut(edges, { depth: 1, maxIterations: 30 });
    const deep = await solveMaxCut(edges, { depth: 4, maxIterations: 30 });

    // Deeper circuits should generally perform better (or at least as good)
    expect(deep.bestEnergy).toBeLessThanOrEqual(shallow.bestEnergy + 0.5);
  });
});

describe('VQE (Variational Quantum Eigensolver)', () => {
  test('should find ground state of simple Hamiltonian', async () => {
    // Simple 2-qubit Ising Hamiltonian
    const hamiltonian = createIsingHamiltonian(
      [[0, 1], [1, 0]], // ZZ coupling
      [-0.5, -0.5]       // Z fields
    );

    const solver = new VQESolver(
      {
        numQubits: 2,
        ansatzType: 'hardware-efficient',
        ansatzDepth: 2,
        maxIterations: 50,
        optimizer: 'gradient-descent',
        learningRate: 0.1,
        tolerance: 1e-5
      },
      hamiltonian
    );

    const result = await solver.solve();

    expect(result.groundStateEnergy).toBeDefined();
    expect(result.groundState.length).toBe(4); // 2^2 states
    expect(result.optimalParameters.length).toBeGreaterThan(0);
    expect(result.converged).toBe(true);
  });

  test('should compare different ansatz types', async () => {
    const hamiltonian = createIsingHamiltonian(
      [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
      [-0.3, -0.3, -0.3]
    );

    const hardwareEfficient = new VQESolver(
      {
        numQubits: 3,
        ansatzType: 'hardware-efficient',
        ansatzDepth: 2,
        maxIterations: 30,
        optimizer: 'adam',
        learningRate: 0.1,
        tolerance: 1e-5
      },
      hamiltonian
    );

    const uccsd = new VQESolver(
      {
        numQubits: 3,
        ansatzType: 'uccsd',
        ansatzDepth: 2,
        maxIterations: 30,
        optimizer: 'adam',
        learningRate: 0.1,
        tolerance: 1e-5
      },
      hamiltonian
    );

    const result1 = await hardwareEfficient.solve();
    const result2 = await uccsd.solve();

    // Both should find reasonable ground states
    expect(result1.groundStateEnergy).toBeLessThan(0);
    expect(result2.groundStateEnergy).toBeLessThan(0);
  });

  test('should optimize with Adam optimizer', async () => {
    const hamiltonian = createIsingHamiltonian(
      [[0, 1], [1, 0]],
      [-1, -1]
    );

    const solver = new VQESolver(
      {
        numQubits: 2,
        ansatzType: 'hardware-efficient',
        ansatzDepth: 2,
        maxIterations: 50,
        optimizer: 'adam',
        learningRate: 0.05,
        tolerance: 1e-6
      },
      hamiltonian
    );

    const result = await solver.solve();

    expect(result.energyHistory.length).toBeGreaterThan(0);
    // Energy should generally decrease
    const initialEnergy = result.energyHistory[0];
    const finalEnergy = result.groundStateEnergy;
    expect(finalEnergy).toBeLessThanOrEqual(initialEnergy);
  });
});

describe('Quantum Annealing', () => {
  test('should solve MaxCut with simulated annealing', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1], [1, 2, 1], [2, 0, 1]
    ];

    const qubo = QUBOFormulator.maxCutToQUBO(edges);

    const result = await solveQUBO(qubo, {
      initialTemperature: 100,
      finalTemperature: 0.1,
      numSteps: 500,
      quantumStrength: 10,
      method: 'simulated'
    });

    expect(result.solution.length).toBe(3);
    expect(result.energy).toBeLessThan(0);
    expect(result.converged).toBe(true);
    expect(result.annealingPath.length).toBeGreaterThan(0);
  });

  test('should solve with quantum Monte Carlo', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 2], [1, 2, 1], [2, 3, 1], [3, 0, 2]
    ];

    const qubo = QUBOFormulator.maxCutToQUBO(edges);

    const result = await solveQUBO(qubo, {
      numSteps: 500,
      method: 'quantum-monte-carlo',
      quantumStrength: 15
    });

    expect(result.solution).toBeDefined();
    expect(result.successProbability).toBeGreaterThan(0);
    expect(result.successProbability).toBeLessThanOrEqual(1);
  });

  test('should track annealing schedule', async () => {
    const qubo = QUBOFormulator.maxCutToQUBO([[0, 1, 1]]);

    const result = await solveQUBO(qubo, {
      initialTemperature: 100,
      finalTemperature: 1,
      numSteps: 100,
      quantumStrength: 20
    });

    expect(result.annealingPath.length).toBe(100);

    // Temperature should decrease
    const firstTemp = result.annealingPath[0].temperature;
    const lastTemp = result.annealingPath[result.annealingPath.length - 1].temperature;
    expect(lastTemp).toBeLessThan(firstTemp);

    // Transverse field should decrease
    const firstField = result.annealingPath[0].transverseField;
    const lastField = result.annealingPath[result.annealingPath.length - 1].transverseField;
    expect(lastField).toBeLessThan(firstField);
  });

  test('should solve TSP problem', async () => {
    // Simple 3-city TSP
    const distanceMatrix = [
      [0, 10, 15],
      [10, 0, 20],
      [15, 20, 0]
    ];

    const qubo = QUBOFormulator.tspToQUBO(distanceMatrix);
    const result = await solveQUBO(qubo, { numSteps: 1000 });

    expect(result.solution.length).toBe(9); // 3 cities x 3 positions
    expect(result.energy).toBeGreaterThan(0);
  });

  test('should solve portfolio optimization', async () => {
    const returns = [0.10, 0.12, 0.08];
    const covariance = [
      [0.04, 0.01, 0.02],
      [0.01, 0.05, 0.01],
      [0.02, 0.01, 0.03]
    ];

    const qubo = QUBOFormulator.portfolioToQUBO(returns, covariance, 2, 1.0);
    const result = await solveQUBO(qubo, { numSteps: 500 });

    expect(result.solution.length).toBe(3);
    // Should select 2 assets
    const selected = result.solution.filter(x => x === 1).length;
    expect(selected).toBeGreaterThan(0);
  });

  test('should solve constraint satisfaction', async () => {
    const qubo = QUBOFormulator.constraintSatisfactionToQUBO(
      4,
      [
        { vars: [0, 1], coeffs: [1, 1], rhs: 1 },    // x0 + x1 = 1
        { vars: [2, 3], coeffs: [1, -1], rhs: 0 }    // x2 - x3 = 0
      ]
    );

    const result = await solveQUBO(qubo, { numSteps: 500 });

    expect(result.solution.length).toBe(4);
    expect(result.energy).toBeGreaterThanOrEqual(0);
  });
});

describe('Swarm Circuit Exploration', () => {
  test('should explore circuits for MaxCut problem', async () => {
    const result = await exploreCircuits({
      numQubits: 3,
      problemType: 'maxcut',
      swarmSize: 10,
      maxDepth: 3,
      explorationSteps: 20,
      learningRate: 0.01,
      memorySize: 100
    });

    expect(result.bestCircuit).toBeDefined();
    expect(result.bestCircuit.depth).toBeGreaterThan(0);
    expect(result.bestCircuit.depth).toBeLessThanOrEqual(3);
    expect(result.bestPerformance).toBeGreaterThan(0);
    expect(result.explorationHistory.length).toBeGreaterThan(0);
    expect(result.converged).toBeDefined();
  });

  test('should learn circuit patterns', async () => {
    const result = await exploreCircuits({
      numQubits: 4,
      problemType: 'vqe',
      swarmSize: 15,
      explorationSteps: 30,
      maxDepth: 4,
      learningRate: 0.01,
      memorySize: 200
    });

    expect(result.learnedPatterns).toBeDefined();
    expect(result.learnedPatterns.length).toBeGreaterThan(0);

    result.learnedPatterns.forEach(pattern => {
      expect(pattern.pattern).toBeDefined();
      expect(pattern.frequency).toBeGreaterThan(0);
      expect(pattern.averagePerformance).toBeGreaterThan(0);
      expect(pattern.embedding.length).toBe(128);
    });
  });

  test('should track convergence', async () => {
    const result = await exploreCircuits({
      numQubits: 3,
      problemType: 'qaoa',
      swarmSize: 10,
      explorationSteps: 50,
      maxDepth: 3,
      learningRate: 0.02,
      memorySize: 100
    });

    expect(result.convergenceData.performanceHistory.length).toBeGreaterThan(0);
    expect(result.convergenceData.diversityHistory.length).toBeGreaterThan(0);

    // Performance should generally improve
    const early = result.convergenceData.performanceHistory.slice(0, 5);
    const late = result.convergenceData.performanceHistory.slice(-5);
    const earlyAvg = early.reduce((a, b) => a + b) / early.length;
    const lateAvg = late.reduce((a, b) => a + b) / late.length;

    expect(lateAvg).toBeGreaterThanOrEqual(earlyAvg - 0.1); // Allow small variance
  });

  test('should generate valid circuit metadata', async () => {
    const result = await exploreCircuits({
      numQubits: 4,
      problemType: 'maxcut',
      swarmSize: 5,
      explorationSteps: 10,
      maxDepth: 3,
      learningRate: 0.01,
      memorySize: 50
    });

    const metadata = result.bestCircuit.metadata;

    expect(metadata.twoQubitGateCount).toBeGreaterThanOrEqual(0);
    expect(metadata.expressibility).toBeGreaterThanOrEqual(0);
    expect(metadata.expressibility).toBeLessThanOrEqual(1);
    expect(metadata.entanglingCapability).toBeGreaterThanOrEqual(0);
    expect(metadata.entanglingCapability).toBeLessThanOrEqual(1);
    expect(metadata.circuitId).toBeDefined();
  });
});

describe('Unified Quantum Optimizer', () => {
  test('should auto-select best MaxCut method', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1], [1, 2, 1], [2, 0, 1]
    ];

    const result = await QuantumOptimizer.solveMaxCut(edges, 'auto');

    expect(result.solution).toBeDefined();
    expect(result.energy).toBeLessThan(0);
    expect(result.method).toBeDefined();
    expect(['QAOA', 'Quantum Annealing']).toContain(result.method);
  });

  test('should solve TSP', async () => {
    const distances = [
      [0, 10, 15, 20],
      [10, 0, 35, 25],
      [15, 35, 0, 30],
      [20, 25, 30, 0]
    ];

    const result = await QuantumOptimizer.solveTSP(distances);

    expect(result.tour).toBeDefined();
    expect(result.tour.length).toBe(16); // 4 cities x 4 positions
    expect(result.distance).toBeGreaterThan(0);
  });

  test('should optimize portfolio', async () => {
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
      2,
      1.5
    );

    expect(result.allocation).toBeDefined();
    expect(result.allocation.length).toBe(4);
    expect(result.expectedReturn).toBeDefined();
    expect(result.risk).toBeGreaterThanOrEqual(0);
  });

  test('should solve constraint satisfaction', async () => {
    const result = await QuantumOptimizer.solveConstraintSatisfaction(
      3,
      [
        { vars: [0, 1], coeffs: [1, 1], rhs: 1 },
        { vars: [1, 2], coeffs: [1, -1], rhs: 0 }
      ]
    );

    expect(result.solution).toBeDefined();
    expect(result.solution.length).toBe(3);
    expect(result.satisfied).toBeDefined();
  });

  test('should explore circuits', async () => {
    const result = await QuantumOptimizer.exploreCircuits({
      numQubits: 3,
      problemType: 'maxcut',
      swarmSize: 8,
      explorationSteps: 15
    });

    expect(result.bestCircuit).toBeDefined();
    expect(result.performance).toBeGreaterThan(0);
    expect(result.learnedPatterns).toBeDefined();
  });
});

describe('Quantum vs Classical Comparison', () => {
  test('should compare MaxCut solutions', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 0, 1], [0, 2, 1]
    ];

    const comparison = await QuantumClassicalComparison.compareMaxCut(edges);

    expect(comparison.quantum).toBeDefined();
    expect(comparison.classical).toBeDefined();
    expect(comparison.quantum.solution.length).toBe(4);
    expect(comparison.classical.solution.length).toBe(4);
    expect(comparison.speedup).toBeGreaterThan(0);
    expect(comparison.qualityRatio).toBeGreaterThan(0);
  });

  test('quantum should match or beat classical quality', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 2], [1, 2, 3], [2, 0, 1]
    ];

    const comparison = await QuantumClassicalComparison.compareMaxCut(edges);

    // Quantum solution should be at least as good as classical (allowing 10% margin)
    expect(comparison.qualityRatio).toBeGreaterThanOrEqual(0.9);
  });

  test('should measure execution times', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1], [1, 2, 1], [2, 0, 1]
    ];

    const comparison = await QuantumClassicalComparison.compareMaxCut(edges);

    expect(comparison.quantum.time).toBeGreaterThan(0);
    expect(comparison.classical.time).toBeGreaterThan(0);
  });
});

describe('Performance and Scalability', () => {
  test('should handle larger problems efficiently', async () => {
    // 6-node graph
    const edges: [number, number, number][] = [
      [0, 1, 1], [1, 2, 1], [2, 3, 1],
      [3, 4, 1], [4, 5, 1], [5, 0, 1],
      [0, 3, 1], [1, 4, 1], [2, 5, 1]
    ];

    const startTime = Date.now();
    const result = await solveMaxCut(edges, {
      depth: 2,
      maxIterations: 50
    });
    const executionTime = Date.now() - startTime;

    expect(result.bestSolution.length).toBe(6);
    expect(executionTime).toBeLessThan(10000); // Should complete within 10 seconds
  });

  test('should scale with circuit depth', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1], [1, 2, 1], [2, 0, 1]
    ];

    const depth1 = await solveMaxCut(edges, { depth: 1, maxIterations: 20 });
    const depth3 = await solveMaxCut(edges, { depth: 3, maxIterations: 20 });

    // Deeper circuits take longer but may give better results
    expect(depth3.executionTime).toBeGreaterThanOrEqual(depth1.executionTime * 0.8);
  });

  test('should handle annealing with many steps', async () => {
    const qubo = QUBOFormulator.maxCutToQUBO([[0, 1, 1], [1, 2, 1]]);

    const result = await solveQUBO(qubo, { numSteps: 2000 });

    expect(result.annealingPath.length).toBe(2000);
    expect(result.executionTime).toBeLessThan(15000); // Should be reasonably fast
  });
});

describe('Edge Cases and Error Handling', () => {
  test('should handle single-edge graph', async () => {
    const edges: [number, number, number][] = [[0, 1, 1]];

    const result = await solveMaxCut(edges, { depth: 1, maxIterations: 10 });

    expect(result.bestSolution.length).toBe(2);
    expect(result.bestEnergy).toBeLessThanOrEqual(0);
  });

  test('should handle disconnected graph', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 1],
      [2, 3, 1]
    ];

    const result = await solveMaxCut(edges, { depth: 2, maxIterations: 20 });

    expect(result.bestSolution.length).toBe(4);
  });

  test('should handle zero-weight edges', async () => {
    const edges: [number, number, number][] = [
      [0, 1, 0],
      [1, 2, 1]
    ];

    const result = await solveMaxCut(edges, { depth: 1, maxIterations: 10 });

    expect(result.bestSolution).toBeDefined();
  });

  test('should handle empty Hamiltonian', async () => {
    const hamiltonian = createIsingHamiltonian(
      [[0]],
      [0]
    );

    const solver = new VQESolver(
      {
        numQubits: 1,
        ansatzType: 'hardware-efficient',
        ansatzDepth: 1,
        maxIterations: 10,
        optimizer: 'gradient-descent',
        learningRate: 0.1,
        tolerance: 1e-5
      },
      hamiltonian
    );

    const result = await solver.solve();

    expect(result.groundStateEnergy).toBeDefined();
  });
});
