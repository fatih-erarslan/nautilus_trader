# @neural-trader/example-quantum-optimization

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-quantum-optimization.svg)](https://www.npmjs.com/package/@neural-trader/example-quantum-optimization)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-quantum-optimization.svg)](https://www.npmjs.com/package/@neural-trader/example-quantum-optimization)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)]()

Quantum-inspired optimization algorithms with swarm-based circuit exploration for combinatorial and constraint problems. Implements QAOA, VQE, Quantum Annealing, and intelligent circuit design using AgentDB for pattern learning.

## Features

- **QAOA (Quantum Approximate Optimization Algorithm)** - Hybrid quantum-classical algorithm for combinatorial problems
- **VQE (Variational Quantum Eigensolver)** - Ground state energy computation and optimization
- **Quantum Annealing Simulation** - Quantum tunneling and thermal annealing for global optimization
- **Swarm Circuit Exploration** - Multi-agent exploration of quantum circuit designs with AgentDB
- **Memory-Based Learning** - Pattern recognition and self-learning optimal circuit depths
- **OpenRouter Integration** - LLM-powered problem decomposition and circuit suggestions
- **Quantum vs Classical Comparison** - Comprehensive benchmarking tools

## Installation

```bash
npm install @neural-trader/example-quantum-optimization
```

## Quick Start

### Solve MaxCut Problem with QAOA

```typescript
import { solveMaxCut } from '@neural-trader/example-quantum-optimization';

// Define graph edges [node1, node2, weight]
const edges: [number, number, number][] = [
  [0, 1, 1],
  [1, 2, 1],
  [2, 3, 1],
  [3, 0, 1],
  [0, 2, 1]
];

const result = await solveMaxCut(edges, {
  depth: 3,           // QAOA circuit depth
  maxIterations: 100,
  learningRate: 0.1,
  tolerance: 1e-6
});

console.log('Best Cut:', result.bestSolution);
console.log('Cut Value:', -result.bestEnergy);
console.log('Converged:', result.converged);
console.log('Time:', result.executionTime, 'ms');
```

### Ground State Energy with VQE

```typescript
import { VQESolver, createIsingHamiltonian } from '@neural-trader/example-quantum-optimization';

// Create Ising Hamiltonian
const hamiltonian = createIsingHamiltonian(
  [[0, 1, 0], [1, 0, 1], [0, 1, 0]], // ZZ couplings
  [-0.5, -0.5, -0.5]                  // Z fields
);

const solver = new VQESolver(
  {
    numQubits: 3,
    ansatzType: 'hardware-efficient',
    ansatzDepth: 3,
    maxIterations: 100,
    optimizer: 'adam',
    learningRate: 0.1,
    tolerance: 1e-6
  },
  hamiltonian
);

const result = await solver.solve();

console.log('Ground State Energy:', result.groundStateEnergy);
console.log('Optimal Parameters:', result.optimalParameters);
console.log('Energy History:', result.energyHistory);
```

### Quantum Annealing for TSP

```typescript
import { solveTSPAnnealing } from '@neural-trader/example-quantum-optimization';

const distanceMatrix = [
  [0, 10, 15, 20],
  [10, 0, 35, 25],
  [15, 35, 0, 30],
  [20, 25, 30, 0]
];

const result = await solveTSPAnnealing(distanceMatrix, {
  initialTemperature: 100,
  finalTemperature: 0.1,
  numSteps: 2000,
  quantumStrength: 15,
  method: 'quantum-monte-carlo'
});

console.log('Best Tour:', result.solution);
console.log('Tour Length:', result.energy);
console.log('Success Probability:', result.successProbability);
console.log('Annealing Path:', result.annealingPath.length, 'steps');
```

### Swarm-Based Circuit Exploration

```typescript
import { exploreCircuits } from '@neural-trader/example-quantum-optimization';

const result = await exploreCircuits({
  numQubits: 4,
  problemType: 'maxcut',
  swarmSize: 20,
  maxDepth: 5,
  explorationSteps: 100,
  learningRate: 0.01,
  memorySize: 1000,
  useOpenRouter: true,
  openrouterApiKey: process.env.OPENROUTER_API_KEY
});

console.log('Best Circuit Depth:', result.bestCircuit.depth);
console.log('Circuit Performance:', result.bestPerformance);
console.log('Learned Patterns:', result.learnedPatterns.length);
console.log('Converged:', result.convergenceData.converged);

// Analyze learned patterns
result.learnedPatterns.forEach(pattern => {
  console.log(`Pattern: ${pattern.pattern}`);
  console.log(`Frequency: ${pattern.frequency}`);
  console.log(`Avg Performance: ${pattern.averagePerformance}`);
});
```

## Unified API

### QuantumOptimizer

High-level API for common quantum optimization tasks:

```typescript
import { QuantumOptimizer } from '@neural-trader/example-quantum-optimization';

// Auto-select best method for MaxCut
const maxcutResult = await QuantumOptimizer.solveMaxCut(edges, 'auto');

// Solve TSP
const tspResult = await QuantumOptimizer.solveTSP(distanceMatrix);

// Portfolio optimization
const portfolioResult = await QuantumOptimizer.optimizePortfolio(
  returns,
  covarianceMatrix,
  budget,
  riskAversion
);

// Constraint satisfaction
const cspResult = await QuantumOptimizer.solveConstraintSatisfaction(
  numVars,
  constraints
);

// Circuit exploration
const circuitResult = await QuantumOptimizer.exploreCircuits({
  numQubits: 4,
  problemType: 'vqe',
  swarmSize: 20,
  explorationSteps: 100
});
```

## Applications

### 1. MaxCut Problem

Find maximum cut in weighted graphs:

```typescript
import { QuantumOptimizer } from '@neural-trader/example-quantum-optimization';

const edges: [number, number, number][] = [
  [0, 1, 5], [0, 2, 3], [1, 2, 2],
  [1, 3, 4], [2, 3, 6], [2, 4, 2],
  [3, 4, 3]
];

const result = await QuantumOptimizer.solveMaxCut(edges, 'qaoa');

// Partition graph into two sets
const set1 = result.solution
  .map((bit, idx) => bit === 0 ? idx : -1)
  .filter(x => x >= 0);
const set2 = result.solution
  .map((bit, idx) => bit === 1 ? idx : -1)
  .filter(x => x >= 0);

console.log('Partition 1:', set1);
console.log('Partition 2:', set2);
console.log('Cut Value:', -result.energy);
```

### 2. Portfolio Optimization

Quantum-inspired portfolio selection:

```typescript
const returns = [0.12, 0.10, 0.08, 0.15, 0.11];
const covarianceMatrix = [
  [0.04, 0.02, 0.01, 0.03, 0.01],
  [0.02, 0.03, 0.01, 0.02, 0.02],
  [0.01, 0.01, 0.02, 0.01, 0.01],
  [0.03, 0.02, 0.01, 0.05, 0.02],
  [0.01, 0.02, 0.01, 0.02, 0.03]
];

const result = await QuantumOptimizer.optimizePortfolio(
  returns,
  covarianceMatrix,
  3,    // Select 3 assets
  1.5   // Risk aversion parameter
);

console.log('Asset Allocation:', result.allocation);
console.log('Expected Return:', (result.expectedReturn * 100).toFixed(2) + '%');
console.log('Portfolio Risk:', (result.risk * 100).toFixed(2) + '%');

// Calculate Sharpe ratio
const riskFreeRate = 0.02;
const sharpe = (result.expectedReturn - riskFreeRate) / result.risk;
console.log('Sharpe Ratio:', sharpe.toFixed(2));
```

### 3. Traveling Salesman Problem

Find shortest tour visiting all cities:

```typescript
const cities = ['A', 'B', 'C', 'D'];
const distances = [
  [0, 10, 15, 20],
  [10, 0, 35, 25],
  [15, 35, 0, 30],
  [20, 25, 30, 0]
];

const result = await QuantumOptimizer.solveTSP(distances);

// Decode tour from solution
const n = cities.length;
const tour: number[] = [];
for (let pos = 0; pos < n; pos++) {
  for (let city = 0; city < n; city++) {
    if (result.tour[pos * n + city] === 1) {
      tour.push(city);
      break;
    }
  }
}

console.log('Tour:', tour.map(i => cities[i]).join(' → '));
console.log('Total Distance:', result.distance);
```

### 4. Constraint Satisfaction

Solve systems of constraints:

```typescript
// Example: Solve x0 + x1 = 1, x1 + x2 = 1, x2 + x3 = 1
const constraints = [
  { vars: [0, 1], coeffs: [1, 1], rhs: 1 },
  { vars: [1, 2], coeffs: [1, 1], rhs: 1 },
  { vars: [2, 3], coeffs: [1, 1], rhs: 1 }
];

const result = await QuantumOptimizer.solveConstraintSatisfaction(4, constraints);

console.log('Solution:', result.solution);
console.log('All Constraints Satisfied:', result.satisfied);

// Verify constraints
constraints.forEach(({ vars, coeffs, rhs }, idx) => {
  const sum = vars.reduce((s, v, i) => s + coeffs[i] * result.solution[v], 0);
  console.log(`Constraint ${idx}: ${sum} = ${rhs} ✓`);
});
```

### 5. Circuit Design Exploration

Discover optimal quantum circuit architectures:

```typescript
const exploration = await exploreCircuits({
  numQubits: 5,
  problemType: 'vqe',
  swarmSize: 30,
  maxDepth: 6,
  explorationSteps: 200,
  learningRate: 0.02,
  memorySize: 2000
});

console.log('Best Circuit:');
console.log('  Depth:', exploration.bestCircuit.depth);
console.log('  Gates:', exploration.bestCircuit.gates.length);
console.log('  Two-Qubit Gates:', exploration.bestCircuit.metadata.twoQubitGateCount);
console.log('  Expressibility:', exploration.bestCircuit.metadata.expressibility.toFixed(3));
console.log('  Entangling Capability:', exploration.bestCircuit.metadata.entanglingCapability.toFixed(3));

// Analyze convergence
const { performanceHistory, diversityHistory } = exploration.convergenceData;
console.log('Initial Performance:', performanceHistory[0].toFixed(3));
console.log('Final Performance:', performanceHistory[performanceHistory.length - 1].toFixed(3));
console.log('Improvement:',
  ((performanceHistory[performanceHistory.length - 1] / performanceHistory[0] - 1) * 100).toFixed(1) + '%'
);
```

## Quantum vs Classical Comparison

Compare quantum-inspired methods against classical algorithms:

```typescript
import { QuantumClassicalComparison } from '@neural-trader/example-quantum-optimization';

const edges: [number, number, number][] = [
  [0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1],
  [4, 0, 1], [0, 2, 1], [1, 3, 1], [2, 4, 1]
];

const comparison = await QuantumClassicalComparison.compareMaxCut(edges);

console.log('Quantum Solution:');
console.log('  Energy:', comparison.quantum.energy);
console.log('  Time:', comparison.quantum.time, 'ms');

console.log('Classical Solution:');
console.log('  Energy:', comparison.classical.energy);
console.log('  Time:', comparison.classical.time, 'ms');

console.log('Performance:');
console.log('  Speedup:', comparison.speedup.toFixed(2) + 'x');
console.log('  Quality Ratio:', comparison.qualityRatio.toFixed(2));
console.log('  Winner:', comparison.qualityRatio > 1 ? 'Quantum' : 'Classical');
```

## Algorithm Details

### QAOA (Quantum Approximate Optimization Algorithm)

Hybrid quantum-classical algorithm that alternates between:
1. **Problem Hamiltonian** (cost function): Encodes optimization objective
2. **Mixer Hamiltonian** (quantum superposition): Explores solution space

**Key Parameters:**
- `depth` (p): Number of QAOA layers (more = better quality, slower)
- `maxIterations`: Classical optimization iterations
- `learningRate`: Parameter update step size
- `tolerance`: Convergence threshold

**Best For:**
- Small to medium graphs (< 20 nodes)
- High-quality solutions needed
- When quantum advantage is expected

### VQE (Variational Quantum Eigensolver)

Finds ground state energies using parameterized quantum circuits:
1. **Ansatz**: Parameterized circuit (hardware-efficient or UCCSD)
2. **Measurement**: Compute expectation value of Hamiltonian
3. **Classical Optimization**: Update parameters to minimize energy

**Ansatz Types:**
- `hardware-efficient`: Alternating rotations and entangling gates
- `uccsd`: Unitary Coupled Cluster (for molecular simulation)

**Best For:**
- Finding ground states
- Molecular simulation
- Optimization through Hamiltonian minimization

### Quantum Annealing

Simulates quantum annealing process:
1. **Initialization**: Start in quantum superposition
2. **Annealing**: Gradually reduce quantum fluctuations
3. **Measurement**: Read out classical solution

**Methods:**
- `simulated`: Classical simulated annealing with quantum tunneling
- `quantum-monte-carlo`: Path integral quantum Monte Carlo
- `path-integral`: Direct path integral formulation

**Best For:**
- Large combinatorial problems
- TSP, scheduling, logistics
- When local minima are problematic

### Swarm Circuit Exploration

Multi-agent exploration of circuit designs:
1. **Swarm Initialization**: Random circuit population
2. **Evaluation**: Test circuit performance
3. **Learning**: Store high-performing patterns in AgentDB
4. **Update**: Move toward better circuits using PSO dynamics
5. **Pattern Recognition**: Extract common successful patterns

**Best For:**
- Automated circuit design
- Finding optimal ansatz for specific problems
- Learning problem-specific circuit structures

## Configuration Options

### QAOA Configuration

```typescript
interface QAOAConfig {
  numQubits: number;        // Problem size
  depth: number;            // QAOA layers (1-10)
  maxIterations: number;    // Classical optimization steps (50-200)
  learningRate: number;     // Parameter updates (0.01-0.5)
  tolerance: number;        // Convergence threshold (1e-6)
}
```

### VQE Configuration

```typescript
interface VQEConfig {
  numQubits: number;
  ansatzType: 'hardware-efficient' | 'uccsd' | 'custom';
  ansatzDepth: number;         // Circuit layers (1-5)
  maxIterations: number;       // Optimization steps (50-200)
  optimizer: 'gradient-descent' | 'adam' | 'cobyla';
  learningRate: number;        // Adam/GD rate (0.01-0.2)
  tolerance: number;           // Convergence (1e-6)
}
```

### Annealing Configuration

```typescript
interface AnnealingConfig {
  numQubits: number;
  initialTemperature: number;  // Start temp (50-200)
  finalTemperature: number;    // End temp (0.01-1)
  annealingTime: number;       // Total time (1000-10000)
  numSteps: number;            // Discrete steps (500-5000)
  quantumStrength: number;     // Transverse field (5-20)
  method: 'simulated' | 'quantum-monte-carlo' | 'path-integral';
}
```

### Circuit Exploration Configuration

```typescript
interface CircuitExplorationConfig {
  numQubits: number;
  problemType: 'maxcut' | 'vqe' | 'qaoa' | 'custom';
  swarmSize: number;          // Number of agents (10-50)
  maxDepth: number;           // Max circuit depth (3-10)
  explorationSteps: number;   // Iterations (50-500)
  learningRate: number;       // Learning rate (0.001-0.1)
  memorySize: number;         // AgentDB size (100-10000)
  useOpenRouter?: boolean;    // Enable LLM assistance
  openrouterApiKey?: string;  // API key
}
```

## Performance Tips

### For Small Problems (< 10 qubits)
- Use QAOA with depth 3-5
- Enable circuit exploration for custom ansatz
- Higher iteration counts (100-200)

### For Large Problems (> 10 qubits)
- Use quantum annealing
- Quantum Monte Carlo method recommended
- More annealing steps (2000-5000)

### For Real-Time Applications
- Reduce depth/iterations
- Use pre-learned circuit patterns
- Enable AgentDB caching

### Memory Optimization
- Enable AgentDB quantization for large pattern sets
- Use streaming for long annealing paths
- Batch circuit evaluations

## Testing

```bash
npm test
```

Run specific test suites:

```bash
npm test -- qaoa
npm test -- vqe
npm test -- annealing
npm test -- swarm
npm test -- comparison
```

## Examples

See the `examples/` directory for complete examples:

- `maxcut.ts` - Graph partitioning
- `portfolio.ts` - Financial optimization
- `tsp.ts` - Routing problems
- `circuits.ts` - Automated circuit design
- `comparison.ts` - Quantum vs classical benchmarks

## API Reference

### Main Classes

- **QAOAOptimizer** - Quantum Approximate Optimization Algorithm
- **VQESolver** - Variational Quantum Eigensolver
- **QuantumAnnealer** - Quantum annealing simulation
- **SwarmCircuitExplorer** - Multi-agent circuit exploration
- **QuantumOptimizer** - Unified high-level API
- **QuantumClassicalComparison** - Benchmarking tools

### Helper Functions

- `createMaxCutProblem()` - Convert edges to QAOA problem
- `createIsingHamiltonian()` - Create Hamiltonian for VQE
- `QUBOFormulator.maxCutToQUBO()` - MaxCut → QUBO
- `QUBOFormulator.tspToQUBO()` - TSP → QUBO
- `QUBOFormulator.portfolioToQUBO()` - Portfolio → QUBO
- `QUBOFormulator.constraintSatisfactionToQUBO()` - CSP → QUBO

## Requirements

- Node.js >= 18.0.0
- Dependencies:
  - `agentdb` - Vector database for pattern learning
  - `mathjs` - Mathematical operations
  - `openai` - Optional LLM integration

## License

MIT

## Contributing

Contributions welcome! Areas of interest:

- Real quantum hardware backends (IBM, Rigetti, IonQ)
- Additional optimization algorithms (ADMM, gradient-free methods)
- More problem formulations (scheduling, protein folding)
- Performance optimizations (GPU acceleration, parallelization)
- Circuit optimization techniques

## References

- **QAOA**: Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
- **VQE**: Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014)
- **Quantum Annealing**: Kadowaki & Nishimori, "Quantum annealing in the transverse Ising model" (1998)
- **Circuit Learning**: Sim et al., "Expressibility and entangling capability of parameterized quantum circuits" (2019)

## Support

For issues, questions, or contributions:
- GitHub Issues: [neural-trader](https://github.com/your-org/neural-trader)
- Documentation: [neural-trader.dev](https://neural-trader.dev)

---

**Note**: This package implements quantum-inspired algorithms using classical simulation. No real quantum hardware is required or used. For production quantum computing, consider integrating with actual quantum backends (IBM Qiskit, AWS Braket, etc.).
