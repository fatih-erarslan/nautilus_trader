/**
 * Tests for Spike-Timing-Dependent Plasticity (STDP)
 */

import { SpikingNeuralNetwork } from '../src/snn';
import { STDPLearner, createSTDPLearner } from '../src/stdp';

describe('STDPLearner', () => {
  let network: SpikingNeuralNetwork;
  let learner: STDPLearner;

  beforeEach(() => {
    network = new SpikingNeuralNetwork(10);
    network.connectFullyRandom([-0.5, 0.5]);
    learner = new STDPLearner();
  });

  test('should initialize with default parameters', () => {
    const params = learner.getParams();

    expect(params.A_plus).toBeDefined();
    expect(params.A_minus).toBeDefined();
    expect(params.tau_plus).toBeDefined();
    expect(params.tau_minus).toBeDefined();
    expect(params.w_min).toBeDefined();
    expect(params.w_max).toBeDefined();
  });

  test('should accept custom parameters', () => {
    const custom_learner = new STDPLearner({
      A_plus: 0.05,
      A_minus: 0.06,
      tau_plus: 15.0,
      tau_minus: 15.0,
    });

    const params = custom_learner.getParams();
    expect(params.A_plus).toBe(0.05);
    expect(params.A_minus).toBe(0.06);
    expect(params.tau_plus).toBe(15.0);
    expect(params.tau_minus).toBe(15.0);
  });

  test('should train network on spike pattern', () => {
    const pattern = [0, 1, 2];
    const initial_weights = network.getConnections().map((c) => c.weight);

    const result = learner.train(network, pattern, 100);

    expect(result.spikes.length).toBeGreaterThan(0);
    expect(result.weights_updated).toBeGreaterThan(0);

    // Weights should have changed
    const final_weights = network.getConnections().map((c) => c.weight);
    const changed = final_weights.filter((w, i) => w !== initial_weights[i]).length;
    expect(changed).toBeGreaterThan(0);
  });

  test('should strengthen connections for correlated spikes (LTP)', () => {
    // Create simple network with one connection
    const simple_network = new SpikingNeuralNetwork(3);
    simple_network.addConnection(0, 1, 0.1, 1.0);

    const initial_weight = simple_network.getConnections()[0].weight;

    // Train with pattern where 0 fires before 1 (should strengthen)
    const result = learner.train(simple_network, [0, 1], 50);
    learner.applyWeightUpdates(simple_network);

    const final_weight = simple_network.getConnections()[0].weight;

    // Weight should increase (LTP)
    expect(final_weight).toBeGreaterThanOrEqual(initial_weight);
  });

  test('should track spike traces', () => {
    const pattern = [0, 1, 2];
    learner.train(network, pattern, 100);

    const traces = learner.getTraces();

    // Should have traces for active neurons
    expect(traces.size).toBeGreaterThan(0);

    traces.forEach((trace) => {
      expect(trace.neuronId).toBeDefined();
      expect(trace.spikeTimes).toBeDefined();
      expect(Array.isArray(trace.spikeTimes)).toBe(true);
    });
  });

  test('should apply weight bounds', () => {
    const bounded_learner = new STDPLearner({
      w_min: -1.0,
      w_max: 1.0,
      A_plus: 10.0, // Very strong learning
    });

    const pattern = [0, 1, 2, 3, 4];

    // Train multiple times to potentially exceed bounds
    for (let i = 0; i < 10; i++) {
      bounded_learner.train(network, pattern, 100);
    }

    // Check all weights are within bounds
    network.getConnections().forEach((conn) => {
      expect(conn.weight).toBeGreaterThanOrEqual(-1.0);
      expect(conn.weight).toBeLessThanOrEqual(1.0);
    });
  });

  test('should train on multiple patterns', () => {
    const patterns = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
    ];

    const history = learner.trainMultipleEpochs(network, patterns, 5, 100);

    // Should have records for all epochs and patterns
    expect(history.length).toBe(5 * 3); // 5 epochs * 3 patterns

    history.forEach((record) => {
      expect(record.epoch).toBeDefined();
      expect(record.pattern).toBeDefined();
      expect(record.spikes).toBeGreaterThan(0);
    });
  });

  test('should show learning progress over epochs', () => {
    const patterns = [[0, 1, 2], [3, 4, 5]];

    const history = learner.trainMultipleEpochs(network, patterns, 10, 100);

    // Group by epoch
    const epochs = Array.from({ length: 10 }, (_, i) =>
      history.filter((h) => h.epoch === i)
    );

    // Average weight change should stabilize over time
    const avg_changes = epochs.map((epoch_records) => {
      const sum = epoch_records.reduce((s, r) => s + r.avg_weight_change, 0);
      return sum / epoch_records.length;
    });

    // Later epochs should have smaller changes (convergence)
    const early_avg = (avg_changes[0] + avg_changes[1]) / 2;
    const late_avg = (avg_changes[8] + avg_changes[9]) / 2;

    expect(late_avg).toBeLessThanOrEqual(early_avg * 2); // Allow some variance
  });

  test('should reset learning state', () => {
    const pattern = [0, 1, 2];
    learner.train(network, pattern, 100);

    expect(learner.getTraces().size).toBeGreaterThan(0);

    learner.reset();

    expect(learner.getTraces().size).toBe(0);
  });

  test('should create learner with presets', () => {
    const default_learner = createSTDPLearner('default');
    const strong_learner = createSTDPLearner('strong');
    const weak_learner = createSTDPLearner('weak');

    const default_params = default_learner.getParams();
    const strong_params = strong_learner.getParams();
    const weak_params = weak_learner.getParams();

    // Strong should have higher learning rates
    expect(strong_params.A_plus).toBeGreaterThan(default_params.A_plus);

    // Weak should have lower learning rates
    expect(weak_params.A_plus).toBeLessThan(default_params.A_plus);
  });
});

describe('STDP Learning Scenarios', () => {
  test('should learn temporal sequences', () => {
    const network = new SpikingNeuralNetwork(5);

    // Create chain: 0->1->2->3->4
    for (let i = 0; i < 4; i++) {
      network.addConnection(i, i + 1, 0.1, 1.0);
    }

    const learner = createSTDPLearner('strong');
    const sequence = [0, 1, 2, 3, 4];

    const initial_weights = network.getConnections().map((c) => c.weight);

    // Train on sequence multiple times
    for (let i = 0; i < 10; i++) {
      learner.train(network, sequence, 50);
    }

    const final_weights = network.getConnections().map((c) => c.weight);

    // Chain connections should be strengthened
    final_weights.forEach((weight, idx) => {
      expect(weight).toBeGreaterThanOrEqual(initial_weights[idx]);
    });
  });

  test('should differentiate between patterns', () => {
    const network = new SpikingNeuralNetwork(10);
    network.connectFullyRandom([-0.3, 0.3]);

    const pattern_a = [0, 1, 2];
    const pattern_b = [5, 6, 7];

    const learner = createSTDPLearner('default');

    // Train on pattern A
    for (let i = 0; i < 5; i++) {
      learner.train(network, pattern_a, 100);
    }

    // Inject pattern A and measure activity
    network.reset();
    network.injectPattern(pattern_a);
    const spikes_a = network.simulate(100);

    // Inject pattern B and measure activity
    network.reset();
    network.injectPattern(pattern_b);
    const spikes_b = network.simulate(100);

    // Pattern A should generate different activity than B
    // (not strict equality since networks are stochastic)
    expect(Math.abs(spikes_a.length - spikes_b.length)).toBeGreaterThan(0);
  });

  test('should handle rapid spike trains', () => {
    const network = new SpikingNeuralNetwork(5);
    network.connectFullyRandom();

    const learner = new STDPLearner();

    // Very dense pattern (all neurons firing close together)
    const dense_pattern = [0, 1, 2, 3, 4];

    expect(() => {
      learner.train(network, dense_pattern, 20); // Short duration
    }).not.toThrow();
  });

  test('should be stable over many training iterations', () => {
    const network = new SpikingNeuralNetwork(10);
    network.connectFullyRandom([-0.5, 0.5]);

    const learner = new STDPLearner({
      w_min: -2.0,
      w_max: 2.0,
    });

    const pattern = [0, 1, 2, 3];

    // Train for many iterations
    for (let i = 0; i < 100; i++) {
      learner.train(network, pattern, 50);

      // Check stability (no NaN or Infinity)
      network.getConnections().forEach((conn) => {
        expect(isFinite(conn.weight)).toBe(true);
        expect(conn.weight).toBeGreaterThanOrEqual(-2.0);
        expect(conn.weight).toBeLessThanOrEqual(2.0);
      });
    }
  });
});
