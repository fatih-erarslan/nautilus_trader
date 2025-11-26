/**
 * @neural-trader/example-logistics-optimization
 *
 * Self-learning vehicle routing optimization with multi-agent swarm coordination
 *
 * Features:
 * - Vehicle Routing Problem (VRP) with time windows
 * - Multi-agent swarm optimization (genetic, simulated annealing, ant colony)
 * - Adaptive learning with AgentDB
 * - Real-time route re-optimization
 * - Traffic pattern learning
 * - OpenRouter for constraint reasoning
 * - Sublinear solver for large-scale instances
 */

export { VRPRouter } from './router';
export { SwarmCoordinator, SwarmAgent, SwarmMessage } from './swarm-coordinator';
export { SelfLearningSystem, MemoryEntry } from './self-learning';

export {
  Location,
  TimeWindow,
  Customer,
  Vehicle,
  Route,
  Solution,
  TrafficPattern,
  OptimizationConfig,
  SwarmConfig,
  LearningMetrics
} from './types';

import { VRPRouter } from './router';
import { SwarmCoordinator } from './swarm-coordinator';
import { SelfLearningSystem } from './self-learning';
import {
  Customer,
  Vehicle,
  SwarmConfig,
  OptimizationConfig,
  Solution,
  LearningMetrics
} from './types';

/**
 * Main logistics optimization system
 */
export class LogisticsOptimizer {
  private router: VRPRouter;
  private swarmCoordinator: SwarmCoordinator | null;
  private learningSystem: SelfLearningSystem;
  private episodeCount: number;

  constructor(
    private customers: Customer[],
    private vehicles: Vehicle[],
    private useSwarm: boolean = true,
    private swarmConfig?: SwarmConfig
  ) {
    this.router = new VRPRouter(customers, vehicles);
    this.learningSystem = new SelfLearningSystem(0.1);
    this.episodeCount = 0;

    if (useSwarm && swarmConfig) {
      this.swarmCoordinator = new SwarmCoordinator(
        swarmConfig,
        customers,
        vehicles
      );
    } else {
      this.swarmCoordinator = null;
    }
  }

  /**
   * Optimize routes using swarm or single-agent
   */
  async optimize(algorithm?: 'genetic' | 'simulated-annealing' | 'ant-colony'): Promise<Solution> {
    const startTime = Date.now();
    let solution: Solution;

    if (this.swarmCoordinator) {
      console.log('Using swarm optimization...');
      solution = await this.swarmCoordinator.optimize();
    } else {
      console.log(`Using single-agent ${algorithm || 'genetic'} optimization...`);
      const config: OptimizationConfig = {
        algorithm: algorithm || 'genetic',
        maxIterations: 100,
        populationSize: 50,
        mutationRate: 0.1,
        crossoverRate: 0.8
      };

      switch (config.algorithm) {
        case 'genetic':
          solution = await this.router.solveGenetic(config);
          break;
        case 'simulated-annealing':
          solution = await this.router.solveSimulatedAnnealing(config);
          break;
        case 'ant-colony':
          solution = await this.router.solveAntColony(config);
          break;
      }
    }

    // Learn from solution
    const metrics: LearningMetrics = {
      episodeId: `episode-${this.episodeCount++}`,
      timestamp: Date.now(),
      solutionQuality: solution.fitness,
      computeTime: Date.now() - startTime,
      customersServed: solution.routes.reduce((sum, r) => sum + r.customers.length, 0)
    };

    await this.learningSystem.learnFromSolution(solution, this.customers, metrics);

    return solution;
  }

  /**
   * Get optimization recommendations using LLM
   */
  async getRecommendations(solution: Solution): Promise<string> {
    if (this.swarmCoordinator) {
      return await this.swarmCoordinator.reasonAboutConstraints(solution);
    }
    return 'Swarm coordinator not available for recommendations.';
  }

  /**
   * Get similar past solutions
   */
  async getSimilarSolutions(topK: number = 5): Promise<any[]> {
    return await this.learningSystem.retrieveSimilarSolutions(
      this.customers.length,
      this.vehicles.length,
      topK
    );
  }

  /**
   * Get learning statistics
   */
  getStatistics() {
    return this.learningSystem.getStatistics();
  }

  /**
   * Get swarm status (if using swarm)
   */
  getSwarmStatus() {
    return this.swarmCoordinator?.getStatus() || null;
  }

  /**
   * Get swarm agents (if using swarm)
   */
  getSwarmAgents() {
    return this.swarmCoordinator?.getAgents() || [];
  }

  /**
   * Export learned patterns
   */
  exportPatterns() {
    return this.learningSystem.exportPatterns();
  }

  /**
   * Import learned patterns
   */
  importPatterns(data: any) {
    this.learningSystem.importPatterns(data);
  }
}

/**
 * Helper function to create sample data
 */
export function createSampleData(numCustomers: number = 50, numVehicles: number = 5): {
  customers: Customer[];
  vehicles: Vehicle[];
} {
  const customers: Customer[] = [];
  const vehicles: Vehicle[] = [];

  // Create customers in a grid pattern
  for (let i = 0; i < numCustomers; i++) {
    const lat = 40.7 + (Math.random() - 0.5) * 0.2; // NYC area
    const lng = -74.0 + (Math.random() - 0.5) * 0.2;

    customers.push({
      id: `customer-${i}`,
      location: {
        id: `loc-customer-${i}`,
        lat,
        lng,
        name: `Customer ${i}`
      },
      demand: Math.floor(Math.random() * 50) + 10,
      timeWindow: {
        start: Date.now() + Math.random() * 3600000, // Within next hour
        end: Date.now() + 7200000 + Math.random() * 3600000 // 2-3 hours from now
      },
      serviceTime: Math.floor(Math.random() * 20) + 5, // 5-25 minutes
      priority: Math.floor(Math.random() * 10) + 1
    });
  }

  // Create vehicles
  const depot: Location = {
    id: 'depot',
    lat: 40.7128,
    lng: -74.0060,
    name: 'Main Depot'
  };

  for (let i = 0; i < numVehicles; i++) {
    vehicles.push({
      id: `vehicle-${i}`,
      capacity: 200,
      startLocation: depot,
      endLocation: depot,
      availableTimeWindow: {
        start: Date.now(),
        end: Date.now() + 28800000 // 8 hours
      },
      costPerKm: 0.5,
      costPerHour: 20,
      maxWorkingHours: 8
    });
  }

  return { customers, vehicles };
}
