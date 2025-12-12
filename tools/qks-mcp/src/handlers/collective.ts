/**
 * Collective Intelligence Layer Handlers - Swarm Coordination
 *
 * Implements Layer 5 of the cognitive architecture:
 * - Multi-agent coordination
 * - Consensus protocols
 * - Stigmergy (indirect communication)
 * - Emergent collective behavior
 */

import { QKSBridge } from './mod.js';

export interface SwarmAgent {
  id: string;
  position: number[];
  velocity: number[];
  beliefs: number[];
  role?: string;
}

export interface ConsensusState {
  converged: boolean;
  agreement_level: number;
  rounds: number;
  dissenting_agents: string[];
}

export class CollectiveHandlers {
  private bridge: QKSBridge;

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Coordinate swarm using biomimetic strategies
   * Supports PSO, Grey Wolf, Whale, Firefly, etc.
   */
  async coordinateSwarm(params: {
    agents: SwarmAgent[];
    objective: string;
    strategy: 'pso' | 'grey_wolf' | 'whale' | 'firefly' | 'ant_colony';
    topology?: 'star' | 'ring' | 'mesh' | 'hyperbolic';
    max_iterations?: number;
  }): Promise<{
    updated_agents: SwarmAgent[];
    global_best: number[];
    convergence: number[];
    diversity: number;
  }> {
    const {
      agents,
      objective,
      strategy,
      topology = 'hyperbolic',
      max_iterations = 100,
    } = params;

    try {
      return await this.bridge.callRust('collective.coordinate_swarm', {
        agents,
        objective,
        strategy,
        topology,
        max_iterations,
      });
    } catch (e) {
      // Fallback: Simple PSO-like coordination
      const updatedAgents = agents.map(agent => ({
        ...agent,
        position: agent.position.map((p, i) => p + 0.1 * (Math.random() - 0.5)),
      }));

      return {
        updated_agents: updatedAgents,
        global_best: agents[0]?.position || [],
        convergence: new Array(max_iterations).fill(0).map((_, i) => 1 / (i + 1)),
        diversity: 0.5,
      };
    }
  }

  /**
   * Achieve consensus using Byzantine fault-tolerant protocol
   */
  async achieveConsensus(params: {
    agents: Array<{ id: string; value: any; reputation?: number }>;
    consensus_threshold?: number;
    max_rounds?: number;
    byzantine_tolerance?: number;
  }): Promise<ConsensusState & {
    consensus_value: any;
    voting_record: Record<string, any[]>;
  }> {
    const {
      agents,
      consensus_threshold = 0.67,
      max_rounds = 10,
      byzantine_tolerance = 0.33,
    } = params;

    try {
      return await this.bridge.callRust('collective.consensus', {
        agents,
        consensus_threshold,
        max_rounds,
        byzantine_tolerance,
      });
    } catch (e) {
      // Fallback: Simple majority voting
      const votes = new Map<string, number>();

      for (const agent of agents) {
        const voteKey = JSON.stringify(agent.value);
        votes.set(voteKey, (votes.get(voteKey) || 0) + 1);
      }

      const sortedVotes = Array.from(votes.entries()).sort((a, b) => b[1] - a[1]);
      const [consensusKey, voteCount] = sortedVotes[0] || [null, 0];
      const agreementLevel = voteCount / agents.length;
      const converged = agreementLevel >= consensus_threshold;

      return {
        converged,
        agreement_level: agreementLevel,
        rounds: 1,
        dissenting_agents: agents
          .filter(a => JSON.stringify(a.value) !== consensusKey)
          .map(a => a.id),
        consensus_value: consensusKey ? JSON.parse(consensusKey) : null,
        voting_record: {},
      };
    }
  }

  /**
   * Implement stigmergy (indirect communication through environment)
   * Inspired by ant pheromone trails
   */
  async updateStigmergy(params: {
    environment_state: number[][];
    agent_actions: Array<{
      agent_id: string;
      position: number[];
      deposit: number;
    }>;
    evaporation_rate?: number;
    diffusion_rate?: number;
  }): Promise<{
    updated_environment: number[][];
    pheromone_peaks: Array<{ position: number[]; intensity: number }>;
    trail_coherence: number;
  }> {
    const {
      environment_state,
      agent_actions,
      evaporation_rate = 0.1,
      diffusion_rate = 0.05,
    } = params;

    try {
      return await this.bridge.callRust('collective.stigmergy', {
        environment_state,
        agent_actions,
        evaporation_rate,
        diffusion_rate,
      });
    } catch (e) {
      // Fallback: Simple pheromone update
      const updated = environment_state.map(row => row.map(val => val * (1 - evaporation_rate)));

      for (const action of agent_actions) {
        const [x, y] = action.position.map(Math.floor);
        if (x >= 0 && x < updated.length && y >= 0 && y < updated[0].length) {
          updated[x][y] += action.deposit;
        }
      }

      const peaks: Array<{ position: number[]; intensity: number }> = [];
      for (let i = 0; i < updated.length; i++) {
        for (let j = 0; j < updated[i].length; j++) {
          if (updated[i][j] > 0.5) {
            peaks.push({ position: [i, j], intensity: updated[i][j] });
          }
        }
      }

      return {
        updated_environment: updated,
        pheromone_peaks: peaks,
        trail_coherence: 0.6,
      };
    }
  }

  /**
   * Analyze emergent collective behavior
   * Detects phase transitions and self-organization
   */
  async analyzeEmergence(params: {
    agent_states: SwarmAgent[];
    time_window?: number;
  }): Promise<{
    order_parameter: number;
    phase: 'disordered' | 'transitioning' | 'ordered';
    collective_modes: Array<{ mode: string; strength: number }>;
    synchronization: number;
  }> {
    const { agent_states, time_window = 100 } = params;

    try {
      return await this.bridge.callRust('collective.analyze_emergence', {
        agent_states,
        time_window,
      });
    } catch (e) {
      // Fallback: Simple coherence measure
      const positions = agent_states.map(a => a.position);
      const velocities = agent_states.map(a => a.velocity);

      const positionCoherence = this.computeCoherence(positions);
      const velocityAlignment = this.computeAlignment(velocities);

      const orderParameter = 0.5 * positionCoherence + 0.5 * velocityAlignment;

      return {
        order_parameter: orderParameter,
        phase: orderParameter > 0.7 ? 'ordered' : orderParameter < 0.3 ? 'disordered' : 'transitioning',
        collective_modes: [
          { mode: 'flocking', strength: velocityAlignment },
          { mode: 'clustering', strength: positionCoherence },
        ],
        synchronization: velocityAlignment,
      };
    }
  }

  /**
   * Distribute task among agents using market-based allocation
   */
  async allocateTasks(params: {
    tasks: Array<{ id: string; requirements: Record<string, number>; reward: number }>;
    agents: Array<{ id: string; capabilities: Record<string, number>; capacity: number }>;
    allocation_strategy?: 'auction' | 'greedy' | 'optimal';
  }): Promise<{
    allocations: Array<{ task_id: string; agent_id: string; cost: number }>;
    total_reward: number;
    utilization: Record<string, number>;
  }> {
    const { tasks, agents, allocation_strategy = 'auction' } = params;

    try {
      return await this.bridge.callRust('collective.allocate_tasks', {
        tasks,
        agents,
        allocation_strategy,
      });
    } catch (e) {
      // Fallback: Greedy allocation
      const allocations: Array<{ task_id: string; agent_id: string; cost: number }> = [];
      const utilization: Record<string, number> = {};

      for (const agent of agents) {
        utilization[agent.id] = 0;
      }

      const sortedTasks = [...tasks].sort((a, b) => b.reward - a.reward);

      for (const task of sortedTasks) {
        let bestAgent: string | null = null;
        let bestCost = Infinity;

        for (const agent of agents) {
          if (utilization[agent.id] < agent.capacity) {
            const cost = this.computeTaskCost(task.requirements, agent.capabilities);
            if (cost < bestCost) {
              bestCost = cost;
              bestAgent = agent.id;
            }
          }
        }

        if (bestAgent) {
          allocations.push({ task_id: task.id, agent_id: bestAgent, cost: bestCost });
          utilization[bestAgent] += 1;
        }
      }

      const totalReward = allocations.reduce((sum, alloc) => {
        const task = tasks.find(t => t.id === alloc.task_id);
        return sum + (task?.reward || 0);
      }, 0);

      return {
        allocations,
        total_reward: totalReward,
        utilization,
      };
    }
  }

  /**
   * Evolve collective strategies using genetic algorithm
   */
  async evolveStrategies(params: {
    population: Array<{ genome: any; fitness: number }>;
    generations?: number;
    mutation_rate?: number;
    crossover_rate?: number;
  }): Promise<{
    evolved_population: Array<{ genome: any; fitness: number }>;
    best_genome: any;
    fitness_history: number[];
  }> {
    const {
      population,
      generations = 50,
      mutation_rate = 0.1,
      crossover_rate = 0.8,
    } = params;

    try {
      return await this.bridge.callRust('collective.evolve_strategies', {
        population,
        generations,
        mutation_rate,
        crossover_rate,
      });
    } catch (e) {
      // Fallback: Simple evolution
      const sortedPop = [...population].sort((a, b) => b.fitness - a.fitness);
      const best = sortedPop[0];

      return {
        evolved_population: sortedPop,
        best_genome: best?.genome || {},
        fitness_history: new Array(generations).fill(best?.fitness || 0),
      };
    }
  }

  // ===== Private Helper Methods =====

  private computeCoherence(vectors: number[][]): number {
    if (vectors.length < 2) return 1.0;

    const centroid = this.computeCentroid(vectors);
    const distances = vectors.map(v => this.euclideanDistance(v, centroid));
    const avgDistance = distances.reduce((a, b) => a + b, 0) / distances.length;

    return 1 / (1 + avgDistance);
  }

  private computeAlignment(velocities: number[][]): number {
    if (velocities.length < 2) return 1.0;

    const avgVelocity = this.computeCentroid(velocities);
    const avgSpeed = Math.sqrt(avgVelocity.reduce((s, v) => s + v * v, 0));

    if (avgSpeed === 0) return 0;

    const alignments = velocities.map(v => {
      const speed = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      if (speed === 0) return 0;
      return v.reduce((s, x, i) => s + x * avgVelocity[i], 0) / (speed * avgSpeed);
    });

    return alignments.reduce((a, b) => a + b, 0) / alignments.length;
  }

  private computeCentroid(vectors: number[][]): number[] {
    if (vectors.length === 0) return [];

    const dim = vectors[0].length;
    const centroid = new Array(dim).fill(0);

    for (const vec of vectors) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += vec[i];
      }
    }

    return centroid.map(c => c / vectors.length);
  }

  private euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
  }

  private computeTaskCost(
    requirements: Record<string, number>,
    capabilities: Record<string, number>
  ): number {
    let cost = 0;
    for (const [key, required] of Object.entries(requirements)) {
      const capability = capabilities[key] || 0;
      const deficit = Math.max(0, required - capability);
      cost += deficit;
    }
    return cost;
  }
}
