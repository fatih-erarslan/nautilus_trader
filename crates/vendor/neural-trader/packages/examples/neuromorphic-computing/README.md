# @neural-trader/example-neuromorphic-computing

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-neuromorphic-computing.svg)](https://www.npmjs.com/package/@neural-trader/example-neuromorphic-computing)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-neuromorphic-computing.svg)](https://www.npmjs.com/package/@neural-trader/example-neuromorphic-computing)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)]()

Neuromorphic computing with Spiking Neural Networks (SNNs), STDP learning, and reservoir computing for ultra-low-power machine learning.

## Overview

This package demonstrates neuromorphic computing principles using event-driven computation, biological learning rules, and swarm-based topology optimization. It's designed for applications requiring temporal processing, pattern recognition, and energy-efficient machine learning.

## Features

### 🧠 Spiking Neural Networks (SNN)
- **Leaky Integrate-and-Fire (LIF) neurons**: Biologically-inspired neuron model with membrane potential dynamics
- **Event-driven computation**: Only active when spikes occur (energy efficient)
- **Temporal dynamics**: Natural handling of time-series and sequential data
- **Refractory period**: Realistic neuron behavior
- **Memory-efficient spike encoding**: Sparse event representation

### 📈 Spike-Timing-Dependent Plasticity (STDP)
- **Hebbian learning**: "Neurons that fire together, wire together"
- **Long-Term Potentiation (LTP)**: Strengthen connections for correlated activity
- **Long-Term Depression (LTD)**: Weaken connections for anti-correlated activity
- **Adaptive synaptic weights**: Self-learning based on spike timing
- **Configurable time windows**: Control learning sensitivity

### 🌊 Liquid State Machines (Reservoir Computing)
- **Fixed random reservoir**: High-dimensional temporal transformation
- **Simple linear readout**: Only readout layer is trained
- **Excellent for temporal patterns**: Natural memory through recurrent dynamics
- **Computationally efficient**: No backpropagation through time
- **Scalable architecture**: Works with varying reservoir sizes

### 🐝 Swarm Topology Optimization
- **Particle Swarm Optimization (PSO)**: Discover optimal network connectivity
- **Task-specific adaptation**: Optimize for your specific problem
- **Sparse network discovery**: Find minimal connections for maximum performance
- **OpenRouter integration**: Architecture optimization guidance
- **AgentDB persistence**: Store and retrieve optimized topologies

## Installation

```bash
npm install @neural-trader/example-neuromorphic-computing
```

## Quick Start

### Pattern Recognition with STDP

```typescript
import {
  SpikingNeuralNetwork,
  createSTDPLearner,
} from '@neural-trader/example-neuromorphic-computing';

// Create network
const network = new SpikingNeuralNetwork(20);
network.connectFullyRandom([-0.5, 0.5]);

// Create STDP learner
const learner = createSTDPLearner('default');

// Training patterns
const patterns = [
  [0, 1, 2, 3, 4], // Pattern A
  [5, 6, 7, 8, 9], // Pattern B
  [10, 11, 12, 13, 14], // Pattern C
];

// Train with STDP
patterns.forEach((pattern) => {
  learner.train(network, pattern, 100);
});

// Test recognition
network.reset();
network.injectPattern(patterns[0]);
const spikes = network.simulate(100);
console.log(`Generated ${spikes.length} spikes`);
```

### Time-Series Classification with LSM

```typescript
import { createLSM } from '@neural-trader/example-neuromorphic-computing';

// Create Liquid State Machine
const lsm = createLSM('medium', 10, 3);

// Prepare training data
const inputs = [
  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  // ... more patterns
];

const targets = [
  [1, 0, 0], // Class 0
  [0, 1, 0], // Class 1
  // ... more labels
];

// Train readout layer
const error = lsm.trainReadout(inputs, targets, 50);
console.log(`Training MSE: ${error.toFixed(4)}`);

// Make predictions
const prediction = lsm.predict(inputs[0], 50);
console.log('Prediction:', prediction);
```

### Topology Optimization with Swarm

```typescript
import {
  SwarmTopologyOptimizer,
  patternRecognitionFitness,
  FitnessTask,
} from '@neural-trader/example-neuromorphic-computing';

// Define optimization task
const task: FitnessTask = {
  inputs: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
  targets: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  evaluate: patternRecognitionFitness,
};

// Create optimizer
const optimizer = new SwarmTopologyOptimizer(10, {
  swarm_size: 20,
  max_iterations: 50,
});

// Optimize topology
const history = optimizer.optimize(task);

console.log(`Best fitness: ${optimizer.getBestFitness()}`);
console.log(`Optimal connections: ${optimizer.getBestTopology().length}`);

// Create optimized network
const optimized_network = optimizer.createOptimizedNetwork();
```

### AgentDB Integration

```typescript
import { NeuromorphicAgent } from '@neural-trader/example-neuromorphic-computing';

// Create agent with AgentDB
const agent = new NeuromorphicAgent('./neuromorphic.db');

// Store network state
await agent.storeNetwork('my_network', network);

// Store STDP learner
await agent.storeSTDP('my_learner', learner);

// Store LSM
await agent.storeLSM('my_lsm', lsm);

// Store optimized topology
await agent.storeTopology('my_topology', optimizer);

// Retrieve stored data
const retrieved = await agent.retrieve('network:my_network');

// Find similar networks
const similar = await agent.findSimilarNetworks(network, 5);

await agent.close();
```

## API Reference

### SpikingNeuralNetwork

```typescript
// Create network
const network = new SpikingNeuralNetwork(num_neurons, neuron_params?);

// Add connections
network.addConnection(source, target, weight, delay);
network.connectFullyRandom(weight_range);

// Inject spikes
network.injectSpike(neuron_id, time?);
network.injectPattern(pattern);

// Simulate
const spikes = network.simulate(duration, dt);

// State management
network.reset();
const state = network.getState();
```

### LIFNeuron

```typescript
const neuron = new LIFNeuron({
  tau_m: 20.0,        // Membrane time constant (ms)
  v_rest: -70.0,      // Resting potential (mV)
  v_threshold: -55.0, // Firing threshold (mV)
  v_reset: -75.0,     // Reset potential (mV)
  t_refrac: 2.0,      // Refractory period (ms)
});

const fired = neuron.update(current_time, input_current, dt);
const potential = neuron.getMembranePotential();
```

### STDPLearner

```typescript
const learner = createSTDPLearner('default' | 'strong' | 'weak');

// Train on single pattern
const result = learner.train(network, pattern, duration);

// Train on multiple patterns
const history = learner.trainMultipleEpochs(
  network,
  patterns,
  epochs,
  duration
);
```

### LiquidStateMachine

```typescript
const lsm = createLSM('small' | 'medium' | 'large', input_size, output_size);

// Process input
const state = lsm.processInput(input, duration);

// Forward pass
const output = lsm.forward(input, duration);

// Train readout
const error = lsm.trainReadout(train_inputs, train_targets, duration);

// Evaluate
const { mse, accuracy } = lsm.evaluate(test_inputs, test_targets, duration);
```

### SwarmTopologyOptimizer

```typescript
const optimizer = new SwarmTopologyOptimizer(network_size, {
  swarm_size: 20,
  max_connections: 100,
  max_iterations: 50,
});

const history = optimizer.optimize(task);
const topology = optimizer.getBestTopology();
const network = optimizer.createOptimizedNetwork();
const json = optimizer.exportTopology();
```

## Applications

### 1. Time-Series Processing
- Stock price prediction
- Sensor data analysis
- Signal processing
- Anomaly detection

### 2. Pattern Recognition
- Image classification (spike-encoded)
- Audio recognition
- Gesture recognition
- Biometric authentication

### 3. Ultra-Low-Power ML
- Edge computing
- IoT devices
- Neuromorphic hardware (Intel Loihi, IBM TrueNorth)
- Battery-powered systems

### 4. Temporal Sequence Learning
- Natural language processing
- Video analysis
- Predictive maintenance
- Financial time-series

### 5. Robotics
- Sensorimotor control
- Real-time decision making
- Adaptive behavior
- Navigation

## Performance Characteristics

### Energy Efficiency
- **Event-driven**: Only active during spikes (sparse computation)
- **Asynchronous**: No global clock synchronization
- **Low precision**: Binary spikes vs. floating-point activations
- **Hardware friendly**: Maps well to neuromorphic chips

### Temporal Processing
- **Native time handling**: No need for time-step unwrapping
- **Long-term dependencies**: Recurrent dynamics provide memory
- **Continuous time**: Natural handling of irregular sampling

### Scalability
- **Sparse connectivity**: O(connections) not O(neurons²)
- **Parallel simulation**: Independent neuron updates
- **Incremental learning**: STDP updates local to connections

## Advanced Examples

### Custom Fitness Function

```typescript
function myCustomFitness(
  network: SpikingNeuralNetwork,
  inputs: number[][],
  targets: number[][]
): number {
  let total_score = 0;

  inputs.forEach((input, idx) => {
    network.reset();
    network.injectPattern(input);
    const spikes = network.simulate(100);

    // Your custom evaluation logic
    const score = evaluateSpikes(spikes, targets[idx]);
    total_score += score;
  });

  return total_score / inputs.length;
}
```

### Custom Neuron Parameters

```typescript
const fast_neuron = new LIFNeuron({
  tau_m: 5.0,          // Fast dynamics
  v_threshold: -60.0,  // Easy to fire
  t_refrac: 1.0,       // Short refractory
});

const slow_neuron = new LIFNeuron({
  tau_m: 40.0,         // Slow dynamics
  v_threshold: -50.0,  // Hard to fire
  t_refrac: 5.0,       // Long refractory
});
```

### Multi-Layer Architecture

```typescript
// Create layers
const input_layer = new SpikingNeuralNetwork(10);
const hidden_layer = new SpikingNeuralNetwork(50);
const output_layer = new SpikingNeuralNetwork(5);

// Connect layers (manually coordinate simulation)
// In production, use a more sophisticated orchestration
```

## Running the Examples

```bash
# Build the package
npm run build

# Run all examples
node dist/index.js

# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Watch mode
npm run test:watch
```

## Testing

The package includes comprehensive tests covering:
- ✅ LIF neuron dynamics
- ✅ Network connectivity
- ✅ Spike propagation
- ✅ STDP learning rules
- ✅ Reservoir computing
- ✅ Topology optimization
- ✅ Pattern recognition tasks

Run tests:
```bash
npm test
```

## Dependencies

- `@neural-trader/agentdb`: Vector database for network state persistence
- `@neural-trader/agentic-flow`: Multi-agent orchestration (planned)

## Performance Tips

1. **Use sparse connectivity**: Dense networks are computationally expensive
2. **Tune simulation timestep**: Smaller dt is more accurate but slower
3. **Batch training**: Process multiple patterns before updating weights
4. **Prune weak connections**: Remove synapses below threshold
5. **Use quantization**: Reduce weight precision for faster inference

## Neuromorphic Hardware

This implementation is designed to map to neuromorphic hardware:

- **Intel Loihi**: 130,000 neurons, 130M synapses
- **IBM TrueNorth**: 1M neurons, 256M synapses
- **BrainScaleS**: Mixed-signal analog/digital
- **SpiNNaker**: ARM-based digital spikes

## Future Enhancements

- [ ] Multi-compartment neuron models
- [ ] Homeostatic plasticity
- [ ] Short-term plasticity (STP)
- [ ] Reward-modulated STDP
- [ ] Convolutional spike layers
- [ ] GPU acceleration
- [ ] NAPI-RS native bindings

## References

1. Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models*
2. Maass, W., Natschläger, T., & Markram, H. (2002). *Real-time computing without stable states: A new framework for neural computation*
3. Bi, G. Q., & Poo, M. M. (1998). *Synaptic modifications in cultured hippocampal neurons*
4. Kennedy, J., & Eberhart, R. (1995). *Particle swarm optimization*

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Author

Neural Trader Team

## Keywords

neuromorphic, spiking-neural-network, snn, stdp, reservoir-computing, liquid-state-machine, event-driven, low-power-ml, temporal-processing, swarm-optimization, agentdb
