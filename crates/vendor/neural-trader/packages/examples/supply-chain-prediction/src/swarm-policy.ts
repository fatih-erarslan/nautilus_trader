/**
 * Swarm-Based Inventory Policy Optimizer
 *
 * Features:
 * - Particle Swarm Optimization for (s,S) policies
 * - Multi-objective optimization (cost vs service level)
 * - Adaptive policy learning
 * - Parallel policy evaluation
 * - Self-learning service level targets
 */

import { AgenticFlow, Agent } from 'agentic-flow';
import { InventoryOptimizer, OptimizerConfig, OptimizationResult } from './inventory-optimizer';
import { DemandForecaster } from './demand-forecaster';

export interface PolicyParticle {
  id: string;
  position: {
    reorderPoint: number;
    orderUpToLevel: number;
    safetyFactor: number;
  };
  velocity: {
    reorderPoint: number;
    orderUpToLevel: number;
    safetyFactor: number;
  };
  fitness: {
    cost: number;
    serviceLevel: number;
    combined: number;
  };
  personalBest: {
    position: PolicyParticle['position'];
    fitness: PolicyParticle['fitness'];
  };
}

export interface SwarmConfig {
  particles: number;
  iterations: number;
  inertia: number;
  cognitive: number;
  social: number;
  bounds: {
    reorderPoint: [number, number];
    orderUpToLevel: [number, number];
    safetyFactor: [number, number];
  };
  objectives: {
    costWeight: number;
    serviceLevelWeight: number;
  };
}

export interface SwarmResult {
  bestPolicy: PolicyParticle['position'];
  bestFitness: PolicyParticle['fitness'];
  convergenceHistory: number[];
  particles: PolicyParticle[];
  iterations: number;
}

export class SwarmPolicyOptimizer {
  private forecaster: DemandForecaster;
  private optimizer: InventoryOptimizer;
  private config: SwarmConfig;
  private swarm: PolicyParticle[];
  private globalBest: PolicyParticle | null;
  private agenticFlow: AgenticFlow;

  constructor(
    forecaster: DemandForecaster,
    optimizer: InventoryOptimizer,
    config: SwarmConfig
  ) {
    this.forecaster = forecaster;
    this.optimizer = optimizer;
    this.config = config;
    this.swarm = [];
    this.globalBest = null;
    this.agenticFlow = new AgenticFlow({
      maxAgents: config.particles,
      topology: 'mesh',
    });
  }

  /**
   * Initialize swarm with random particles
   */
  private initializeSwarm(): void {
    this.swarm = [];

    for (let i = 0; i < this.config.particles; i++) {
      const particle: PolicyParticle = {
        id: `particle_${i}`,
        position: {
          reorderPoint: this.randomInRange(this.config.bounds.reorderPoint),
          orderUpToLevel: this.randomInRange(this.config.bounds.orderUpToLevel),
          safetyFactor: this.randomInRange(this.config.bounds.safetyFactor),
        },
        velocity: {
          reorderPoint: 0,
          orderUpToLevel: 0,
          safetyFactor: 0,
        },
        fitness: {
          cost: Infinity,
          serviceLevel: 0,
          combined: Infinity,
        },
        personalBest: {
          position: { reorderPoint: 0, orderUpToLevel: 0, safetyFactor: 0 },
          fitness: { cost: Infinity, serviceLevel: 0, combined: Infinity },
        },
      };

      this.swarm.push(particle);
    }
  }

  /**
   * Optimize inventory policies using swarm intelligence
   */
  async optimize(productId: string, currentFeatures: any): Promise<SwarmResult> {
    this.initializeSwarm();

    const convergenceHistory: number[] = [];

    // Main PSO loop
    for (let iteration = 0; iteration < this.config.iterations; iteration++) {
      // Evaluate fitness of all particles in parallel
      await this.evaluateSwarm(productId, currentFeatures);

      // Update personal and global bests
      this.updateBests();

      // Update velocities and positions
      this.updateSwarm();

      // Track convergence
      const avgFitness =
        this.swarm.reduce((sum, p) => sum + p.fitness.combined, 0) / this.swarm.length;
      convergenceHistory.push(avgFitness);

      // Log progress
      console.log(
        `Iteration ${iteration + 1}/${this.config.iterations}: ` +
          `Best Fitness = ${this.globalBest?.fitness.combined.toFixed(2)}, ` +
          `Avg Fitness = ${avgFitness.toFixed(2)}`
      );
    }

    return {
      bestPolicy: this.globalBest!.position,
      bestFitness: this.globalBest!.fitness,
      convergenceHistory,
      particles: this.swarm,
      iterations: this.config.iterations,
    };
  }

  /**
   * Evaluate fitness of all particles in parallel using agentic-flow
   */
  private async evaluateSwarm(productId: string, currentFeatures: any): Promise<void> {
    // Create agents for parallel evaluation
    const agents: Agent[] = this.swarm.map((particle) => ({
      id: particle.id,
      role: 'policy-evaluator',
      task: `Evaluate policy: ${JSON.stringify(particle.position)}`,
      execute: async () => {
        return this.evaluateParticle(particle, productId, currentFeatures);
      },
    }));

    // Execute in parallel
    await this.agenticFlow.executeParallel(agents);
  }

  /**
   * Evaluate fitness of single particle
   */
  private async evaluateParticle(
    particle: PolicyParticle,
    productId: string,
    currentFeatures: any
  ): Promise<void> {
    // Create temporary optimizer config with particle's parameters
    const tempConfig: OptimizerConfig = {
      targetServiceLevel: 0.95,
      planningHorizon: 30,
      reviewPeriod: 7,
      safetyFactor: particle.position.safetyFactor,
      costWeights: {
        holding: 1,
        ordering: 1,
        shortage: 1,
      },
    };

    // Create temporary optimizer
    const tempOptimizer = new InventoryOptimizer(this.forecaster, tempConfig);

    // Copy network from main optimizer
    const topology = this.optimizer.getNetworkTopology();
    for (const node of topology.nodes) {
      tempOptimizer.addNode(node);
    }

    // Simulate with particle's policy
    const simulation = await tempOptimizer.simulate(productId, currentFeatures, 30);

    // Calculate fitness
    const costFitness = simulation.avgInventoryCost;
    const serviceLevelFitness = 1 - simulation.avgServiceLevel; // Minimize (1 - service level)

    // Combined fitness (weighted sum)
    const combinedFitness =
      this.config.objectives.costWeight * costFitness +
      this.config.objectives.serviceLevelWeight * serviceLevelFitness;

    // Update particle fitness
    particle.fitness = {
      cost: costFitness,
      serviceLevel: simulation.avgServiceLevel,
      combined: combinedFitness,
    };
  }

  /**
   * Update personal and global bests
   */
  private updateBests(): void {
    for (const particle of this.swarm) {
      // Update personal best
      if (particle.fitness.combined < particle.personalBest.fitness.combined) {
        particle.personalBest = {
          position: { ...particle.position },
          fitness: { ...particle.fitness },
        };
      }

      // Update global best
      if (!this.globalBest || particle.fitness.combined < this.globalBest.fitness.combined) {
        this.globalBest = {
          ...particle,
          position: { ...particle.position },
          fitness: { ...particle.fitness },
        };
      }
    }
  }

  /**
   * Update velocities and positions using PSO update rules
   */
  private updateSwarm(): void {
    for (const particle of this.swarm) {
      // Update velocity for each dimension
      for (const dim of ['reorderPoint', 'orderUpToLevel', 'safetyFactor'] as const) {
        const r1 = Math.random();
        const r2 = Math.random();

        // PSO velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        particle.velocity[dim] =
          this.config.inertia * particle.velocity[dim] +
          this.config.cognitive * r1 * (particle.personalBest.position[dim] - particle.position[dim]) +
          this.config.social * r2 * (this.globalBest!.position[dim] - particle.position[dim]);

        // Update position
        particle.position[dim] += particle.velocity[dim];

        // Apply bounds
        const bounds = this.config.bounds[dim];
        particle.position[dim] = Math.max(bounds[0], Math.min(bounds[1], particle.position[dim]));
      }
    }
  }

  /**
   * Random value in range
   */
  private randomInRange(range: [number, number]): number {
    return range[0] + Math.random() * (range[1] - range[0]);
  }

  /**
   * Get Pareto front for multi-objective optimization
   */
  getParetoFront(): PolicyParticle[] {
    const front: PolicyParticle[] = [];

    for (const particle of this.swarm) {
      let isDominated = false;

      for (const other of this.swarm) {
        if (particle === other) continue;

        // Check if 'other' dominates 'particle'
        if (
          other.fitness.cost <= particle.fitness.cost &&
          other.fitness.serviceLevel >= particle.fitness.serviceLevel &&
          (other.fitness.cost < particle.fitness.cost ||
            other.fitness.serviceLevel > particle.fitness.serviceLevel)
        ) {
          isDominated = true;
          break;
        }
      }

      if (!isDominated) {
        front.push(particle);
      }
    }

    return front;
  }

  /**
   * Adaptive learning of service level targets
   */
  async adaptServiceLevel(
    productId: string,
    currentFeatures: any,
    targetRevenue: number
  ): Promise<number> {
    // Binary search for optimal service level
    let low = 0.8;
    let high = 0.99;
    let optimalLevel = 0.95;

    while (high - low > 0.01) {
      const mid = (low + high) / 2;

      // Test service level
      const config: OptimizerConfig = {
        targetServiceLevel: mid,
        planningHorizon: 30,
        reviewPeriod: 7,
        safetyFactor: 1.65,
        costWeights: { holding: 1, ordering: 1, shortage: 1 },
      };

      const tempOptimizer = new InventoryOptimizer(this.forecaster, config);

      // Copy network
      const topology = this.optimizer.getNetworkTopology();
      for (const node of topology.nodes) {
        tempOptimizer.addNode(node);
      }

      // Simulate
      const simulation = await tempOptimizer.simulate(productId, currentFeatures, 30);

      // Calculate revenue (simplified)
      const revenue = simulation.fillRate * targetRevenue;
      const profit = revenue - simulation.avgInventoryCost;

      // Adjust search range
      if (profit > targetRevenue * 0.1) {
        // Target 10% margin
        high = mid;
      } else {
        low = mid;
      }

      optimalLevel = mid;
    }

    return optimalLevel;
  }

  /**
   * Export best policy for deployment
   */
  exportPolicy(): {
    policy: PolicyParticle['position'];
    performance: PolicyParticle['fitness'];
    timestamp: number;
  } {
    if (!this.globalBest) {
      throw new Error('No policy available. Run optimization first.');
    }

    return {
      policy: this.globalBest.position,
      performance: this.globalBest.fitness,
      timestamp: Date.now(),
    };
  }
}
