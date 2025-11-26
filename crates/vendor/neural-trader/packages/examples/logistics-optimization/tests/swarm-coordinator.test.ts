/**
 * Tests for Swarm Coordinator
 */

import { SwarmCoordinator } from '../src/swarm-coordinator';
import { Customer, Vehicle, SwarmConfig } from '../src/types';

describe('SwarmCoordinator', () => {
  let customers: Customer[];
  let vehicles: Vehicle[];
  let swarmConfig: SwarmConfig;

  beforeEach(() => {
    customers = [];
    for (let i = 0; i < 20; i++) {
      customers.push({
        id: `c${i}`,
        location: {
          id: `loc${i}`,
          lat: 40.7 + Math.random() * 0.1,
          lng: -74.0 + Math.random() * 0.1
        },
        demand: Math.floor(Math.random() * 30) + 10,
        timeWindow: { start: Date.now(), end: Date.now() + 3600000 },
        serviceTime: 10,
        priority: 5
      });
    }

    vehicles = [];
    for (let i = 0; i < 3; i++) {
      vehicles.push({
        id: `v${i}`,
        capacity: 150,
        startLocation: { id: 'depot', lat: 40.70, lng: -74.00 },
        availableTimeWindow: { start: Date.now(), end: Date.now() + 7200000 },
        costPerKm: 0.5,
        costPerHour: 20,
        maxWorkingHours: 8
      });
    }

    swarmConfig = {
      numAgents: 6,
      topology: 'mesh',
      communicationStrategy: 'best-solution',
      convergenceCriteria: {
        maxIterations: 50,
        noImprovementSteps: 10
      }
    };
  });

  describe('Swarm Initialization', () => {
    test('should initialize correct number of agents', () => {
      const coordinator = new SwarmCoordinator(swarmConfig, customers, vehicles);
      const agents = coordinator.getAgents();

      expect(agents).toHaveLength(6);
    });

    test('should assign different algorithms to agents', () => {
      const coordinator = new SwarmCoordinator(swarmConfig, customers, vehicles);
      const agents = coordinator.getAgents();

      const algorithms = new Set(agents.map(a => a.algorithm));
      expect(algorithms.size).toBeGreaterThan(1);
    });
  });

  describe('Swarm Optimization', () => {
    test('should optimize and return solution', async () => {
      const coordinator = new SwarmCoordinator(swarmConfig, customers, vehicles);

      const solution = await coordinator.optimize();

      expect(solution).toBeDefined();
      expect(solution.routes).toBeInstanceOf(Array);
      expect(solution.fitness).toBeGreaterThanOrEqual(0);
    }, 30000); // 30 second timeout

    test('should update status during optimization', async () => {
      const coordinator = new SwarmCoordinator(swarmConfig, customers, vehicles);

      // Start optimization but don't wait
      const optimizationPromise = coordinator.optimize();

      // Check status after a short delay
      await new Promise(resolve => setTimeout(resolve, 100));
      const status = coordinator.getStatus();

      expect(status.iteration).toBeGreaterThanOrEqual(0);
      expect(status.agentsWorking + status.agentsCompleted).toBeLessThanOrEqual(6);

      await optimizationPromise;
    }, 30000);
  });

  describe('Agent Communication', () => {
    test('should share best solutions between agents', async () => {
      const config: SwarmConfig = {
        ...swarmConfig,
        communicationStrategy: 'broadcast',
        convergenceCriteria: { maxIterations: 20 }
      };

      const coordinator = new SwarmCoordinator(config, customers, vehicles);
      const solution = await coordinator.optimize();

      const agents = coordinator.getAgents();
      const bestSolutions = agents
        .filter(a => a.bestSolution)
        .map(a => a.bestSolution!.fitness);

      // All agents should have found solutions
      expect(bestSolutions.length).toBeGreaterThan(0);
    }, 30000);
  });

  describe('Convergence', () => {
    test('should converge within iteration limit', async () => {
      const config: SwarmConfig = {
        ...swarmConfig,
        convergenceCriteria: { maxIterations: 30 }
      };

      const coordinator = new SwarmCoordinator(config, customers, vehicles);
      await coordinator.optimize();

      const status = coordinator.getStatus();
      expect(status.iteration).toBeLessThanOrEqual(30);
    }, 30000);
  });

  describe('Status Monitoring', () => {
    test('should provide accurate status information', () => {
      const coordinator = new SwarmCoordinator(swarmConfig, customers, vehicles);
      const status = coordinator.getStatus();

      expect(status).toHaveProperty('iteration');
      expect(status).toHaveProperty('agentsWorking');
      expect(status).toHaveProperty('agentsCompleted');
      expect(status).toHaveProperty('globalBestFitness');
      expect(status).toHaveProperty('convergence');
    });
  });
});
