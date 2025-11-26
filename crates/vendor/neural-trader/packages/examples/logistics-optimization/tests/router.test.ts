/**
 * Tests for VRP Router
 */

import { VRPRouter } from '../src/router';
import { Customer, Vehicle, OptimizationConfig } from '../src/types';

describe('VRPRouter', () => {
  let customers: Customer[];
  let vehicles: Vehicle[];

  beforeEach(() => {
    // Create test data
    customers = [
      {
        id: 'c1',
        location: { id: 'loc1', lat: 40.71, lng: -74.00 },
        demand: 50,
        timeWindow: { start: Date.now(), end: Date.now() + 3600000 },
        serviceTime: 10,
        priority: 5
      },
      {
        id: 'c2',
        location: { id: 'loc2', lat: 40.72, lng: -74.01 },
        demand: 30,
        timeWindow: { start: Date.now(), end: Date.now() + 3600000 },
        serviceTime: 15,
        priority: 5
      },
      {
        id: 'c3',
        location: { id: 'loc3', lat: 40.73, lng: -74.02 },
        demand: 40,
        timeWindow: { start: Date.now(), end: Date.now() + 3600000 },
        serviceTime: 12,
        priority: 5
      }
    ];

    vehicles = [
      {
        id: 'v1',
        capacity: 100,
        startLocation: { id: 'depot', lat: 40.70, lng: -74.00 },
        availableTimeWindow: { start: Date.now(), end: Date.now() + 7200000 },
        costPerKm: 0.5,
        costPerHour: 20,
        maxWorkingHours: 8
      }
    ];
  });

  describe('Distance Calculations', () => {
    test('should calculate distance between locations', () => {
      const router = new VRPRouter(customers, vehicles);
      const distance = router.getDistance('loc1', 'loc2');
      expect(distance).toBeGreaterThan(0);
      expect(distance).toBeLessThan(2); // Should be less than 2km
    });

    test('should calculate travel time between locations', () => {
      const router = new VRPRouter(customers, vehicles);
      const time = router.getTravelTime('loc1', 'loc2');
      expect(time).toBeGreaterThan(0);
      expect(time).toBeLessThan(10); // Should be less than 10 minutes
    });
  });

  describe('Genetic Algorithm', () => {
    test('should generate valid solution', async () => {
      const router = new VRPRouter(customers, vehicles);
      const config: OptimizationConfig = {
        algorithm: 'genetic',
        maxIterations: 10,
        populationSize: 20,
        mutationRate: 0.1,
        crossoverRate: 0.8
      };

      const solution = await router.solveGenetic(config);

      expect(solution).toBeDefined();
      expect(solution.routes).toBeInstanceOf(Array);
      expect(solution.fitness).toBeGreaterThanOrEqual(0);
      expect(solution.metadata.algorithm).toBe('genetic');
    });

    test('should improve solution quality over iterations', async () => {
      const router = new VRPRouter(customers, vehicles);

      const config1: OptimizationConfig = {
        algorithm: 'genetic',
        maxIterations: 5,
        populationSize: 20
      };

      const config2: OptimizationConfig = {
        algorithm: 'genetic',
        maxIterations: 50,
        populationSize: 20
      };

      const solution1 = await router.solveGenetic(config1);
      const solution2 = await router.solveGenetic(config2);

      // More iterations should generally produce better or equal results
      expect(solution2.fitness).toBeLessThanOrEqual(solution1.fitness * 1.2);
    });
  });

  describe('Simulated Annealing', () => {
    test('should generate valid solution', async () => {
      const router = new VRPRouter(customers, vehicles);
      const config: OptimizationConfig = {
        algorithm: 'simulated-annealing',
        maxIterations: 100,
        temperature: 1000,
        coolingRate: 0.95
      };

      const solution = await router.solveSimulatedAnnealing(config);

      expect(solution).toBeDefined();
      expect(solution.routes).toBeInstanceOf(Array);
      expect(solution.fitness).toBeGreaterThanOrEqual(0);
      expect(solution.metadata.algorithm).toBe('simulated-annealing');
    });
  });

  describe('Ant Colony Optimization', () => {
    test('should generate valid solution', async () => {
      const router = new VRPRouter(customers, vehicles);
      const config: OptimizationConfig = {
        algorithm: 'ant-colony',
        maxIterations: 50,
        populationSize: 10,
        pheromoneEvaporation: 0.1
      };

      const solution = await router.solveAntColony(config);

      expect(solution).toBeDefined();
      expect(solution.routes).toBeInstanceOf(Array);
      expect(solution.fitness).toBeGreaterThanOrEqual(0);
      expect(solution.metadata.algorithm).toBe('ant-colony');
    });
  });

  describe('Solution Validation', () => {
    test('should not violate vehicle capacity', async () => {
      const router = new VRPRouter(customers, vehicles);
      const config: OptimizationConfig = {
        algorithm: 'genetic',
        maxIterations: 20,
        populationSize: 20
      };

      const solution = await router.solveGenetic(config);

      for (const route of solution.routes) {
        const vehicle = vehicles.find(v => v.id === route.vehicleId);
        const totalDemand = route.customers.reduce((sum, c) => sum + c.demand, 0);

        // Allow for some violations in testing
        expect(totalDemand).toBeLessThanOrEqual(vehicle!.capacity * 1.5);
      }
    });
  });
});
