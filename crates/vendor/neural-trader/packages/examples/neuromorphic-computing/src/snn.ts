/**
 * Spiking Neural Network (SNN) with Leaky Integrate-and-Fire (LIF) neurons
 *
 * LIF neurons integrate incoming spikes and fire when threshold is reached.
 * Features:
 * - Event-driven computation (only active when spikes occur)
 * - Temporal dynamics with membrane potential decay
 * - Refractory period after firing
 * - Memory-efficient spike encoding
 */

export interface LIFNeuronParams {
  /** Membrane time constant (ms) - controls leak rate */
  tau_m: number;
  /** Resting potential (mV) */
  v_rest: number;
  /** Firing threshold (mV) */
  v_threshold: number;
  /** Reset potential after spike (mV) */
  v_reset: number;
  /** Refractory period (ms) - time after spike when neuron cannot fire */
  t_refrac: number;
}

export interface SynapticConnection {
  /** Source neuron index */
  source: number;
  /** Target neuron index */
  target: number;
  /** Synaptic weight */
  weight: number;
  /** Synaptic delay (ms) */
  delay: number;
}

export interface SpikeEvent {
  /** Neuron index that fired */
  neuronId: number;
  /** Time of spike (ms) */
  time: number;
}

/**
 * Leaky Integrate-and-Fire neuron model
 */
export class LIFNeuron {
  private params: LIFNeuronParams;
  private membrane_potential: number;
  private last_spike_time: number;
  private refractory_end: number;

  constructor(params: Partial<LIFNeuronParams> = {}) {
    // Default parameters based on biological neurons
    this.params = {
      tau_m: params.tau_m ?? 20.0, // 20ms membrane time constant
      v_rest: params.v_rest ?? -70.0, // -70mV resting potential
      v_threshold: params.v_threshold ?? -55.0, // -55mV threshold
      v_reset: params.v_reset ?? -75.0, // -75mV reset potential
      t_refrac: params.t_refrac ?? 2.0, // 2ms refractory period
    };

    this.membrane_potential = this.params.v_rest;
    this.last_spike_time = -Infinity;
    this.refractory_end = -Infinity;
  }

  /**
   * Update neuron state for a given timestep
   * @param current_time Current simulation time (ms)
   * @param input_current Input current from synapses (pA)
   * @param dt Time step (ms)
   * @returns True if neuron fired a spike
   */
  update(current_time: number, input_current: number, dt: number): boolean {
    // Check if in refractory period
    if (current_time < this.refractory_end) {
      this.membrane_potential = this.params.v_reset;
      return false;
    }

    // Leaky integration: dV/dt = (v_rest - V + R*I) / tau_m
    const leak = (this.params.v_rest - this.membrane_potential) / this.params.tau_m;
    const dv = (leak + input_current) * dt;
    this.membrane_potential += dv;

    // Check for spike
    if (this.membrane_potential >= this.params.v_threshold) {
      this.membrane_potential = this.params.v_reset;
      this.last_spike_time = current_time;
      this.refractory_end = current_time + this.params.t_refrac;
      return true;
    }

    return false;
  }

  /**
   * Get current membrane potential
   */
  getMembranePotential(): number {
    return this.membrane_potential;
  }

  /**
   * Get time of last spike
   */
  getLastSpikeTime(): number {
    return this.last_spike_time;
  }

  /**
   * Reset neuron to resting state
   */
  reset(): void {
    this.membrane_potential = this.params.v_rest;
    this.last_spike_time = -Infinity;
    this.refractory_end = -Infinity;
  }

  /**
   * Get neuron parameters
   */
  getParams(): LIFNeuronParams {
    return { ...this.params };
  }
}

/**
 * Spiking Neural Network with event-driven computation
 */
export class SpikingNeuralNetwork {
  private neurons: LIFNeuron[];
  private connections: SynapticConnection[];
  private spike_queue: SpikeEvent[];
  private current_time: number;

  constructor(
    num_neurons: number,
    neuron_params?: Partial<LIFNeuronParams>
  ) {
    this.neurons = Array.from(
      { length: num_neurons },
      () => new LIFNeuron(neuron_params)
    );
    this.connections = [];
    this.spike_queue = [];
    this.current_time = 0;
  }

  /**
   * Add synaptic connection between neurons
   */
  addConnection(source: number, target: number, weight: number, delay: number = 1.0): void {
    if (source < 0 || source >= this.neurons.length) {
      throw new Error(`Invalid source neuron: ${source}`);
    }
    if (target < 0 || target >= this.neurons.length) {
      throw new Error(`Invalid target neuron: ${target}`);
    }

    this.connections.push({ source, target, weight, delay });
  }

  /**
   * Create fully connected network with random weights
   */
  connectFullyRandom(weight_range: [number, number] = [-1.0, 1.0]): void {
    const n = this.neurons.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const weight =
            weight_range[0] + Math.random() * (weight_range[1] - weight_range[0]);
          this.addConnection(i, j, weight, 1.0);
        }
      }
    }
  }

  /**
   * Inject spike into specific neuron
   */
  injectSpike(neuron_id: number, time?: number): void {
    const spike_time = time ?? this.current_time;
    this.spike_queue.push({ neuronId: neuron_id, time: spike_time });
    this.spike_queue.sort((a, b) => a.time - b.time);
  }

  /**
   * Inject spike pattern (e.g., input encoding)
   */
  injectPattern(pattern: number[], time?: number): void {
    pattern.forEach((neuron_id, idx) => {
      this.injectSpike(neuron_id, (time ?? this.current_time) + idx * 0.1);
    });
  }

  /**
   * Simulate network for given duration
   * @param duration Simulation duration (ms)
   * @param dt Time step (ms)
   * @returns Array of spike events
   */
  simulate(duration: number, dt: number = 0.1): SpikeEvent[] {
    const output_spikes: SpikeEvent[] = [];
    const end_time = this.current_time + duration;

    while (this.current_time < end_time) {
      // Calculate input currents for each neuron from delayed spikes
      const input_currents = new Array(this.neurons.length).fill(0);

      // Process spikes that should arrive at current time
      const active_spikes = this.spike_queue.filter(
        (spike) => spike.time <= this.current_time
      );

      active_spikes.forEach((spike) => {
        // Find all outgoing connections from spiking neuron
        this.connections
          .filter((conn) => conn.source === spike.neuronId)
          .forEach((conn) => {
            const arrival_time = spike.time + conn.delay;
            if (
              arrival_time <= this.current_time &&
              arrival_time > this.current_time - dt
            ) {
              input_currents[conn.target] += conn.weight;
            }
          });
      });

      // Remove processed spikes
      this.spike_queue = this.spike_queue.filter(
        (spike) => spike.time > this.current_time
      );

      // Update all neurons
      this.neurons.forEach((neuron, idx) => {
        const fired = neuron.update(this.current_time, input_currents[idx], dt);
        if (fired) {
          const spike_event = { neuronId: idx, time: this.current_time };
          output_spikes.push(spike_event);
          this.spike_queue.push(spike_event);
        }
      });

      this.current_time += dt;
    }

    return output_spikes;
  }

  /**
   * Get network state for persistence
   */
  getState(): {
    neurons: Array<{ potential: number; lastSpike: number }>;
    connections: SynapticConnection[];
    time: number;
  } {
    return {
      neurons: this.neurons.map((n) => ({
        potential: n.getMembranePotential(),
        lastSpike: n.getLastSpikeTime(),
      })),
      connections: this.connections,
      time: this.current_time,
    };
  }

  /**
   * Reset network to initial state
   */
  reset(): void {
    this.neurons.forEach((n) => n.reset());
    this.spike_queue = [];
    this.current_time = 0;
  }

  /**
   * Get neuron by index
   */
  getNeuron(index: number): LIFNeuron {
    return this.neurons[index];
  }

  /**
   * Get all connections
   */
  getConnections(): SynapticConnection[] {
    return [...this.connections];
  }

  /**
   * Get number of neurons
   */
  size(): number {
    return this.neurons.length;
  }
}
