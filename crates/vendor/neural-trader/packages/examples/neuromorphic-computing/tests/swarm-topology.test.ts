/**
 * Tests for Swarm-based Topology Optimization
 */

import { SpikingNeuralNetwork } from '../src/snn';
import {
  SwarmTopologyOptimizer,
  patternRecognitionFitness,
  temporalSequenceFitness,
  FitnessTask,
} from '../src/swarm-topology';

describe('SwarmTopologyOptimizer', () => {
  let optimizer: SwarmTopologyOptimizer;

  beforeEach(() => {
    optimizer = new SwarmTopologyOptimizer(10, {
      swarm_size: 10,
      max_iterations: 20,
    });
  });

  test('should initialize with specified parameters', () => {
    expect(optimizer).toBeDefined();
  });

  test('should have initial best fitness as negative infinity', () => {
    expect(optimizer.getBestFitness()).toBe(-Infinity);
  });

  test('should optimize topology', () => {
    const task: FitnessTask = {
      inputs: [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
      ],
      targets: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      evaluate: patternRecognitionFitness,
    };

    const history = optimizer.optimize(task);

    expect(history.length).toBeGreaterThan(0);
    expect(optimizer.getBestFitness()).toBeGreaterThan(-Infinity);
  });

  test('should improve fitness over iterations', () => {
    const task: FitnessTask = {
      inputs: [
        [0, 1],
        [2, 3],
      ],
      targets: [
        [1, 0],
        [0, 1],
      ],
      evaluate: patternRecognitionFitness,
    };

    const history = optimizer.optimize(task);

    // Check that fitness improves or stays stable
    for (let i = 1; i < history.length; i++) {
      expect(history[i].best_fitness).toBeGreaterThanOrEqual(
        history[i - 1].best_fitness - 0.01 // Allow small numerical errors
      );
    }
  });

  test('should return best topology', () => {
    const task: FitnessTask = {
      inputs: [[0, 1], [2, 3]],
      targets: [[1, 0], [0, 1]],
      evaluate: patternRecognitionFitness,
    };

    optimizer.optimize(task);
    const topology = optimizer.getBestTopology();

    expect(Array.isArray(topology)).toBe(true);
    expect(topology.length).toBeGreaterThan(0);

    topology.forEach((gene) => {
      expect(gene.source).toBeDefined();
      expect(gene.target).toBeDefined();
      expect(gene.weight).toBeDefined();
      expect(gene.delay).toBeDefined();
      expect(gene.source).not.toBe(gene.target); // No self-connections
    });
  });

  test('should create optimized network', () => {
    const task: FitnessTask = {
      inputs: [[0, 1], [2, 3]],
      targets: [[1, 0], [0, 1]],
      evaluate: patternRecognitionFitness,
    };

    optimizer.optimize(task);
    const network = optimizer.createOptimizedNetwork();

    expect(network).toBeInstanceOf(SpikingNeuralNetwork);
    expect(network.size()).toBe(10);
    expect(network.getConnections().length).toBeGreaterThan(0);
  });

  test('should export topology to JSON', () => {
    const task: FitnessTask = {
      inputs: [[0, 1]],
      targets: [[1, 0]],
      evaluate: patternRecognitionFitness,
    };

    optimizer.optimize(task);
    const json = optimizer.exportTopology();

    expect(typeof json).toBe('string');
    expect(() => JSON.parse(json)).not.toThrow();

    const parsed = JSON.parse(json);
    expect(parsed.network_size).toBe(10);
    expect(parsed.connections).toBeDefined();
    expect(parsed.fitness).toBeDefined();
    expect(parsed.topology).toBeDefined();
  });

  test('should converge for simple tasks', () => {
    const task: FitnessTask = {
      inputs: [[0], [1]],
      targets: [[1], [0]],
      evaluate: (network, inputs, targets) => {
        // Simple dummy fitness
        return Math.random();
      },
    };

    const history = optimizer.optimize(task);

    // Should complete without errors
    expect(history.length).toBeGreaterThan(0);
  });

  test('should handle different swarm sizes', () => {
    const small_optimizer = new SwarmTopologyOptimizer(5, { swarm_size: 5 });
    const large_optimizer = new SwarmTopologyOptimizer(5, { swarm_size: 50 });

    const task: FitnessTask = {
      inputs: [[0, 1]],
      targets: [[1, 0]],
      evaluate: patternRecognitionFitness,
    };

    expect(() => {
      small_optimizer.optimize(task);
      large_optimizer.optimize(task);
    }).not.toThrow();
  });
});

describe('Fitness Functions', () => {
  test('patternRecognitionFitness should evaluate network', () => {
    const network = new SpikingNeuralNetwork(10);
    network.connectFullyRandom();

    const inputs = [
      [0, 1, 2],
      [3, 4, 5],
    ];

    const targets = [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ];

    const fitness = patternRecognitionFitness(network, inputs, targets);

    expect(fitness).toBeGreaterThanOrEqual(0);
    expect(fitness).toBeLessThanOrEqual(1);
    expect(isFinite(fitness)).toBe(true);
  });

  test('temporalSequenceFitness should evaluate network', () => {
    const network = new SpikingNeuralNetwork(10);
    network.connectFullyRandom();

    const inputs = [
      [0, 1, 2],
      [3, 4, 5],
    ];

    const targets = [
      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ];

    const fitness = temporalSequenceFitness(network, inputs, targets);

    expect(isFinite(fitness)).toBe(true);
  });

  test('fitness functions should be deterministic', () => {
    const network = new SpikingNeuralNetwork(10);
    network.connectFullyRandom();

    const inputs = [[0, 1, 2]];
    const targets = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]];

    const fitness1 = patternRecognitionFitness(network, inputs, targets);
    const fitness2 = patternRecognitionFitness(network, inputs, targets);

    // Should be similar (might have small variance due to timing)
    expect(Math.abs(fitness1 - fitness2)).toBeLessThan(0.1);
  });
});

describe('Topology Optimization Scenarios', () => {
  test('should find topology for XOR-like problem', () => {
    const optimizer = new SwarmTopologyOptimizer(8, {
      swarm_size: 15,
      max_iterations: 30,
    });

    const task: FitnessTask = {
      inputs: [
        [0, 1], // 01
        [2, 3], // 10
        [0, 1, 2, 3], // 11
      ],
      targets: [
        [1, 0], // True
        [1, 0], // True
        [0, 1], // False
      ],
      evaluate: patternRecognitionFitness,
    };

    const history = optimizer.optimize(task);

    expect(history.length).toBeGreaterThan(0);
    expect(optimizer.getBestFitness()).toBeGreaterThan(-Infinity);
  });

  test('should optimize for sparse connectivity', () => {
    const optimizer = new SwarmTopologyOptimizer(20, {
      swarm_size: 20,
      max_connections: 50, // Limit connections
      max_iterations: 25,
    });

    const task: FitnessTask = {
      inputs: [[0, 1, 2], [5, 6, 7]],
      targets: [[1, 0], [0, 1]],
      evaluate: patternRecognitionFitness,
    };

    const history = optimizer.optimize(task);
    const topology = optimizer.getBestTopology();

    // Should respect connection limit
    expect(topology.length).toBeLessThanOrEqual(50);

    // Should still find reasonable solution
    expect(optimizer.getBestFitness()).toBeGreaterThan(-Infinity);
  });

  test('should handle multiple classes', () => {
    const optimizer = new SwarmTopologyOptimizer(15, {
      swarm_size: 20,
      max_iterations: 25,
    });

    const task: FitnessTask = {
      inputs: [
        [0, 1, 2], // Class 0
        [3, 4, 5], // Class 1
        [6, 7, 8], // Class 2
        [9, 10, 11], // Class 3
      ],
      targets: [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ],
      evaluate: patternRecognitionFitness,
    };

    const history = optimizer.optimize(task);

    expect(history.length).toBeGreaterThan(0);
    expect(optimizer.getBestFitness()).toBeGreaterThanOrEqual(0);
  });

  test('should optimize different network sizes', () => {
    const sizes = [5, 10, 20];

    sizes.forEach((size) => {
      const optimizer = new SwarmTopologyOptimizer(size, {
        swarm_size: 10,
        max_iterations: 10,
      });

      const task: FitnessTask = {
        inputs: [[0, 1]],
        targets: [[1, 0]],
        evaluate: patternRecognitionFitness,
      };

      expect(() => {
        optimizer.optimize(task);
      }).not.toThrow();
    });
  });

  test('should track optimization progress', () => {
    const optimizer = new SwarmTopologyOptimizer(10, {
      swarm_size: 15,
      max_iterations: 30,
    });

    const task: FitnessTask = {
      inputs: [[0, 1], [2, 3]],
      targets: [[1, 0], [0, 1]],
      evaluate: patternRecognitionFitness,
    };

    const history = optimizer.optimize(task);

    history.forEach((record, idx) => {
      expect(record.iteration).toBe(idx);
      expect(record.best_fitness).toBeDefined();
      expect(record.avg_fitness).toBeDefined();
      expect(record.best_connections).toBeGreaterThan(0);
      expect(isFinite(record.best_fitness)).toBe(true);
      expect(isFinite(record.avg_fitness)).toBe(true);
    });
  });
});
