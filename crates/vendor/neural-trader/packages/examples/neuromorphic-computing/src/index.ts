/**
 * @neural-trader/example-neuromorphic-computing
 *
 * Neuromorphic computing with Spiking Neural Networks (SNNs), STDP learning,
 * and reservoir computing for ultra-low-power machine learning.
 *
 * Features:
 * - Leaky Integrate-and-Fire (LIF) neurons
 * - Spike-Timing-Dependent Plasticity (STDP) learning
 * - Liquid State Machines (reservoir computing)
 * - Event-driven computation
 * - Swarm-based topology optimization
 * - AgentDB integration for persistent network state
 */

import { AgentDB } from '@neural-trader/agentdb';
import {
  SpikingNeuralNetwork,
  LIFNeuron,
  LIFNeuronParams,
  SynapticConnection,
  SpikeEvent,
} from './snn';
import {
  STDPLearner,
  STDPParams,
  createSTDPLearner,
  SpikeTrace,
} from './stdp';
import {
  LiquidStateMachine,
  ReservoirParams,
  ReadoutWeights,
  createLSM,
} from './reservoir-computing';
import {
  SwarmTopologyOptimizer,
  TopologyGene,
  TopologyParticle,
  SwarmParams,
  FitnessTask,
  patternRecognitionFitness,
  temporalSequenceFitness,
  getOpenRouterArchitectureSuggestion,
} from './swarm-topology';

// Re-export all components
export {
  // SNN
  SpikingNeuralNetwork,
  LIFNeuron,
  LIFNeuronParams,
  SynapticConnection,
  SpikeEvent,

  // STDP
  STDPLearner,
  STDPParams,
  createSTDPLearner,
  SpikeTrace,

  // Reservoir Computing
  LiquidStateMachine,
  ReservoirParams,
  ReadoutWeights,
  createLSM,

  // Swarm Topology
  SwarmTopologyOptimizer,
  TopologyGene,
  TopologyParticle,
  SwarmParams,
  FitnessTask,
  patternRecognitionFitness,
  temporalSequenceFitness,
  getOpenRouterArchitectureSuggestion,
};

/**
 * AgentDB integration for persistent network state
 */
export class NeuromorphicAgent {
  private db: AgentDB;
  private namespace: string;

  constructor(db_path: string = './neuromorphic.db', namespace: string = 'neuromorphic') {
    this.db = new AgentDB(db_path);
    this.namespace = namespace;
  }

  /**
   * Store SNN state in AgentDB
   */
  async storeNetwork(network_id: string, network: SpikingNeuralNetwork): Promise<void> {
    const state = network.getState();

    await this.db.store(
      `${this.namespace}:network:${network_id}`,
      JSON.stringify({
        neurons: state.neurons,
        connections: state.connections,
        time: state.time,
        timestamp: Date.now(),
      })
    );
  }

  /**
   * Store STDP learner state
   */
  async storeSTDP(learner_id: string, learner: STDPLearner): Promise<void> {
    const params = learner.getParams();
    const traces = Array.from(learner.getTraces().entries());

    await this.db.store(
      `${this.namespace}:stdp:${learner_id}`,
      JSON.stringify({
        params,
        traces,
        timestamp: Date.now(),
      })
    );
  }

  /**
   * Store LSM state
   */
  async storeLSM(lsm_id: string, lsm: LiquidStateMachine): Promise<void> {
    const params = lsm.getParams();
    const state = lsm.getReservoirState();
    const weights = lsm.getReadoutWeights();

    await this.db.store(
      `${this.namespace}:lsm:${lsm_id}`,
      JSON.stringify({
        params,
        state,
        weights,
        timestamp: Date.now(),
      })
    );
  }

  /**
   * Store optimized topology
   */
  async storeTopology(topology_id: string, optimizer: SwarmTopologyOptimizer): Promise<void> {
    const topology = optimizer.getBestTopology();
    const fitness = optimizer.getBestFitness();

    await this.db.store(
      `${this.namespace}:topology:${topology_id}`,
      JSON.stringify({
        topology,
        fitness,
        connections: topology.length,
        timestamp: Date.now(),
      })
    );
  }

  /**
   * Retrieve stored data
   */
  async retrieve(key: string): Promise<any> {
    const data = await this.db.retrieve(`${this.namespace}:${key}`);
    return data ? JSON.parse(data) : null;
  }

  /**
   * Search for similar network states using vector embeddings
   */
  async findSimilarNetworks(network: SpikingNeuralNetwork, limit: number = 5): Promise<Array<{ id: string; similarity: number }>> {
    const state = network.getState();

    // Create vector embedding from network state
    const embedding = this.createNetworkEmbedding(state);

    // Search in AgentDB (assuming vector search capability)
    const results = await this.db.search(this.namespace, embedding, limit);

    return results.map((result: any) => ({
      id: result.key,
      similarity: result.score,
    }));
  }

  /**
   * Create vector embedding from network state
   */
  private createNetworkEmbedding(state: any): number[] {
    // Simple embedding: concatenate neuron potentials
    const potentials = state.neurons.map((n: any) => n.potential);

    // Normalize to unit vector
    const magnitude = Math.sqrt(potentials.reduce((sum: number, p: number) => sum + p * p, 0));
    return potentials.map((p: number) => p / (magnitude || 1));
  }

  /**
   * Close database connection
   */
  async close(): Promise<void> {
    await this.db.close();
  }
}

/**
 * High-level example: Pattern recognition with STDP learning
 */
export async function runPatternRecognitionExample(): Promise<void> {
  console.log('=== Neuromorphic Pattern Recognition Example ===\n');

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

  // Train on patterns
  console.log('Training network with STDP...');
  patterns.forEach((pattern, idx) => {
    console.log(`  Pattern ${idx}: [${pattern.join(', ')}]`);
    learner.train(network, pattern, 100);
  });

  // Test recognition
  console.log('\nTesting pattern recognition:');
  patterns.forEach((pattern, idx) => {
    network.reset();
    network.injectPattern(pattern);
    const spikes = network.simulate(100);
    console.log(`  Pattern ${idx}: ${spikes.length} spikes generated`);
  });

  // Store in AgentDB
  const agent = new NeuromorphicAgent();
  await agent.storeNetwork('pattern_recognition', network);
  await agent.storeSTDP('pattern_learning', learner);
  await agent.close();

  console.log('\nNetwork state saved to AgentDB');
}

/**
 * High-level example: Reservoir computing for time-series
 */
export async function runReservoirComputingExample(): Promise<void> {
  console.log('=== Reservoir Computing Example ===\n');

  // Create LSM
  const lsm = createLSM('medium', 10, 3);

  // Generate synthetic time-series data
  const n_samples = 50;
  const inputs: number[][] = [];
  const targets: number[][] = [];

  for (let i = 0; i < n_samples; i++) {
    // Input: random binary pattern
    const input = Array.from({ length: 10 }, () => Math.random() > 0.5 ? 1 : 0);

    // Target: sum of inputs (quantized to 3 classes)
    const sum = input.reduce((a, b) => a + b, 0);
    const target = [0, 0, 0];
    if (sum < 4) target[0] = 1;
    else if (sum < 7) target[1] = 1;
    else target[2] = 1;

    inputs.push(input);
    targets.push(target);
  }

  // Split train/test
  const train_inputs = inputs.slice(0, 40);
  const train_targets = targets.slice(0, 40);
  const test_inputs = inputs.slice(40);
  const test_targets = targets.slice(40);

  // Train readout layer
  console.log('Training reservoir readout layer...');
  const train_error = lsm.trainReadout(train_inputs, train_targets, 50);
  console.log(`  Training MSE: ${train_error.toFixed(4)}`);

  // Evaluate
  const { mse, accuracy } = lsm.evaluate(test_inputs, test_targets, 50);
  console.log(`  Test MSE: ${mse.toFixed(4)}`);
  console.log(`  Test Accuracy: ${(accuracy * 100).toFixed(2)}%`);

  // Store in AgentDB
  const agent = new NeuromorphicAgent();
  await agent.storeLSM('time_series_classifier', lsm);
  await agent.close();

  console.log('\nLSM state saved to AgentDB');
}

/**
 * High-level example: Swarm topology optimization
 */
export async function runSwarmOptimizationExample(): Promise<void> {
  console.log('=== Swarm Topology Optimization Example ===\n');

  // Define pattern recognition task
  const patterns = [
    [0, 1, 2], // Class 0
    [3, 4, 5], // Class 1
    [6, 7, 8], // Class 2
  ];

  const targets = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  const task: FitnessTask = {
    inputs: patterns,
    targets,
    evaluate: patternRecognitionFitness,
  };

  // Create optimizer
  const optimizer = new SwarmTopologyOptimizer(10, {
    swarm_size: 15,
    max_iterations: 30,
  });

  console.log('Optimizing network topology with particle swarm...');
  const history = optimizer.optimize(task);

  // Show progress
  console.log('\nOptimization progress:');
  history.slice(0, 5).forEach((record) => {
    console.log(
      `  Iteration ${record.iteration}: fitness=${record.best_fitness.toFixed(3)}, ` +
      `connections=${record.best_connections}, avg_fitness=${record.avg_fitness.toFixed(3)}`
    );
  });
  console.log('  ...');
  const last = history[history.length - 1];
  console.log(
    `  Iteration ${last.iteration}: fitness=${last.best_fitness.toFixed(3)}, ` +
    `connections=${last.best_connections}, avg_fitness=${last.avg_fitness.toFixed(3)}`
  );

  console.log(`\nBest fitness: ${optimizer.getBestFitness().toFixed(3)}`);
  console.log(`Optimal connections: ${optimizer.getBestTopology().length}`);

  // Store in AgentDB
  const agent = new NeuromorphicAgent();
  await agent.storeTopology('optimized_pattern_classifier', optimizer);
  await agent.close();

  console.log('\nOptimized topology saved to AgentDB');
}

/**
 * Main orchestration function
 */
export async function main(): Promise<void> {
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║  @neural-trader/example-neuromorphic-computing            ║');
  console.log('║  Spiking Neural Networks & Neuromorphic Computing         ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  try {
    // Run all examples
    await runPatternRecognitionExample();
    console.log('\n' + '='.repeat(60) + '\n');

    await runReservoirComputingExample();
    console.log('\n' + '='.repeat(60) + '\n');

    await runSwarmOptimizationExample();

    console.log('\n✓ All examples completed successfully!');
  } catch (error) {
    console.error('Error running examples:', error);
    throw error;
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}
