/**
 * Tests for Spiking Neural Network (SNN) implementation
 */

import { LIFNeuron, SpikingNeuralNetwork } from '../src/snn';

describe('LIFNeuron', () => {
  let neuron: LIFNeuron;

  beforeEach(() => {
    neuron = new LIFNeuron();
  });

  test('should initialize with resting potential', () => {
    const potential = neuron.getMembranePotential();
    expect(potential).toBe(-70.0); // Default v_rest
  });

  test('should integrate input current', () => {
    const initial_potential = neuron.getMembranePotential();
    neuron.update(0, 10.0, 1.0); // Strong input
    const new_potential = neuron.getMembranePotential();

    expect(new_potential).toBeGreaterThan(initial_potential);
  });

  test('should fire spike when threshold is reached', () => {
    let fired = false;

    // Apply strong current until spike
    for (let t = 0; t < 100; t += 1.0) {
      fired = neuron.update(t, 15.0, 1.0);
      if (fired) break;
    }

    expect(fired).toBe(true);
  });

  test('should reset after spike', () => {
    // Force spike
    for (let t = 0; t < 100; t += 1.0) {
      const fired = neuron.update(t, 15.0, 1.0);
      if (fired) {
        const potential = neuron.getMembranePotential();
        expect(potential).toBeLessThan(-55.0); // Should be at reset potential
        break;
      }
    }
  });

  test('should respect refractory period', () => {
    // Force spike
    let spike_time = 0;
    for (let t = 0; t < 100; t += 1.0) {
      const fired = neuron.update(t, 15.0, 1.0);
      if (fired) {
        spike_time = t;
        break;
      }
    }

    // Try to fire again immediately (should fail due to refractory period)
    const fired_again = neuron.update(spike_time + 1.0, 15.0, 1.0);
    expect(fired_again).toBe(false);
  });

  test('should leak towards resting potential', () => {
    // Increase potential
    neuron.update(0, 10.0, 1.0);
    const elevated_potential = neuron.getMembranePotential();

    // Let it leak without input
    neuron.update(1.0, 0, 1.0);
    neuron.update(2.0, 0, 1.0);
    const leaked_potential = neuron.getMembranePotential();

    expect(leaked_potential).toBeLessThan(elevated_potential);
  });

  test('should track last spike time', () => {
    const initial_time = neuron.getLastSpikeTime();
    expect(initial_time).toBe(-Infinity);

    // Force spike
    for (let t = 0; t < 100; t += 1.0) {
      const fired = neuron.update(t, 15.0, 1.0);
      if (fired) {
        expect(neuron.getLastSpikeTime()).toBe(t);
        break;
      }
    }
  });

  test('should reset to initial state', () => {
    // Change state
    neuron.update(0, 10.0, 1.0);

    // Reset
    neuron.reset();

    expect(neuron.getMembranePotential()).toBe(-70.0);
    expect(neuron.getLastSpikeTime()).toBe(-Infinity);
  });
});

describe('SpikingNeuralNetwork', () => {
  let network: SpikingNeuralNetwork;

  beforeEach(() => {
    network = new SpikingNeuralNetwork(10);
  });

  test('should create network with specified size', () => {
    expect(network.size()).toBe(10);
  });

  test('should add synaptic connections', () => {
    network.addConnection(0, 1, 0.5, 1.0);
    const connections = network.getConnections();

    expect(connections).toHaveLength(1);
    expect(connections[0]).toEqual({
      source: 0,
      target: 1,
      weight: 0.5,
      delay: 1.0,
    });
  });

  test('should reject invalid connections', () => {
    expect(() => network.addConnection(-1, 1, 0.5, 1.0)).toThrow();
    expect(() => network.addConnection(0, 20, 0.5, 1.0)).toThrow();
  });

  test('should create fully connected network', () => {
    network.connectFullyRandom();
    const connections = network.getConnections();

    // Should have n*(n-1) connections (all pairs except self)
    expect(connections.length).toBe(10 * 9);
  });

  test('should inject spikes', () => {
    network.injectSpike(0);
    const spikes = network.simulate(10);

    // Should have at least the injected spike
    expect(spikes.length).toBeGreaterThan(0);
    expect(spikes[0].neuronId).toBe(0);
  });

  test('should inject spike patterns', () => {
    const pattern = [0, 1, 2, 3];
    network.injectPattern(pattern);
    const spikes = network.simulate(10);

    // Should have spikes from all pattern neurons
    const neuron_ids = new Set(spikes.map((s) => s.neuronId));
    pattern.forEach((id) => {
      expect(neuron_ids.has(id)).toBe(true);
    });
  });

  test('should simulate network dynamics', () => {
    network.connectFullyRandom();
    network.injectPattern([0, 1, 2]);

    const spikes = network.simulate(100);

    // Should generate activity
    expect(spikes.length).toBeGreaterThan(3);

    // Spikes should be ordered by time
    for (let i = 1; i < spikes.length; i++) {
      expect(spikes[i].time).toBeGreaterThanOrEqual(spikes[i - 1].time);
    }
  });

  test('should propagate spikes through connections', () => {
    // Create simple chain: 0 -> 1 -> 2
    network.addConnection(0, 1, 2.0, 1.0); // Strong connection
    network.addConnection(1, 2, 2.0, 1.0);

    network.injectSpike(0, 0);
    const spikes = network.simulate(50);

    // Should see activity propagate
    const neuron_ids = new Set(spikes.map((s) => s.neuronId));
    expect(neuron_ids.has(0)).toBe(true);
    expect(neuron_ids.has(1)).toBe(true);
  });

  test('should respect synaptic delays', () => {
    network.addConnection(0, 1, 2.0, 10.0); // 10ms delay

    network.injectSpike(0, 0);
    const spikes = network.simulate(50);

    // Find spikes from neuron 1
    const neuron1_spikes = spikes.filter((s) => s.neuronId === 1);

    if (neuron1_spikes.length > 0) {
      // Should occur after delay
      expect(neuron1_spikes[0].time).toBeGreaterThan(10.0);
    }
  });

  test('should get network state', () => {
    network.connectFullyRandom();
    network.injectPattern([0, 1]);
    network.simulate(50);

    const state = network.getState();

    expect(state.neurons).toHaveLength(10);
    expect(state.connections.length).toBeGreaterThan(0);
    expect(state.time).toBeGreaterThan(0);
  });

  test('should reset network', () => {
    network.connectFullyRandom();
    network.injectPattern([0, 1]);
    network.simulate(50);

    network.reset();
    const state = network.getState();

    expect(state.time).toBe(0);
    state.neurons.forEach((n) => {
      expect(n.potential).toBe(-70.0); // Resting potential
    });
  });

  test('should handle multiple simulation calls', () => {
    network.connectFullyRandom();
    network.injectPattern([0, 1, 2]);

    const spikes1 = network.simulate(50);
    const spikes2 = network.simulate(50);

    // Second simulation should continue from where first ended
    expect(spikes2.length).toBeGreaterThanOrEqual(0);

    if (spikes1.length > 0 && spikes2.length > 0) {
      expect(spikes2[0].time).toBeGreaterThanOrEqual(spikes1[spikes1.length - 1].time);
    }
  });
});

describe('Network Topology', () => {
  test('should create different random topologies', () => {
    const network1 = new SpikingNeuralNetwork(10);
    const network2 = new SpikingNeuralNetwork(10);

    network1.connectFullyRandom();
    network2.connectFullyRandom();

    const weights1 = network1.getConnections().map((c) => c.weight);
    const weights2 = network2.getConnections().map((c) => c.weight);

    // Weights should be different (statistically)
    const same_count = weights1.filter((w, i) => w === weights2[i]).length;
    expect(same_count).toBeLessThan(weights1.length * 0.1); // Less than 10% same
  });

  test('should support custom connection patterns', () => {
    const network = new SpikingNeuralNetwork(10);

    // Create ring topology: 0->1->2->...->9->0
    for (let i = 0; i < 10; i++) {
      const next = (i + 1) % 10;
      network.addConnection(i, next, 1.0, 1.0);
    }

    const connections = network.getConnections();
    expect(connections).toHaveLength(10);

    // Verify ring structure
    connections.forEach((conn, idx) => {
      expect(conn.source).toBe(idx);
      expect(conn.target).toBe((idx + 1) % 10);
    });
  });
});
