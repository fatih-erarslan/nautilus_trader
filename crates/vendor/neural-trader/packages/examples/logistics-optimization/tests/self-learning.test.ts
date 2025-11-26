/**
 * Tests for Self-Learning System
 */

import { SelfLearningSystem } from '../src/self-learning';
import { Solution, LearningMetrics, Customer } from '../src/types';

describe('SelfLearningSystem', () => {
  let learningSystem: SelfLearningSystem;
  let mockSolution: Solution;
  let mockCustomers: Customer[];

  beforeEach(() => {
    learningSystem = new SelfLearningSystem(0.1);

    mockCustomers = [
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
      }
    ];

    mockSolution = {
      routes: [
        {
          vehicleId: 'v1',
          customers: mockCustomers,
          totalDistance: 10,
          totalTime: 60,
          totalCost: 50,
          utilizationRate: 0.8,
          timeWindowViolations: 0,
          capacityViolations: 0
        }
      ],
      totalCost: 50,
      totalDistance: 10,
      unassignedCustomers: [],
      fitness: 100,
      metadata: { algorithm: 'genetic', iterations: 50, computeTime: 1000 }
    };
  });

  describe('Learning from Solutions', () => {
    test('should store solution in memory', async () => {
      const metrics: LearningMetrics = {
        episodeId: 'ep1',
        timestamp: Date.now(),
        solutionQuality: 100,
        computeTime: 1000,
        customersServed: 2
      };

      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);

      const stats = learningSystem.getStatistics();
      expect(stats.totalEpisodes).toBe(1);
    });

    test('should track multiple episodes', async () => {
      for (let i = 0; i < 5; i++) {
        const metrics: LearningMetrics = {
          episodeId: `ep${i}`,
          timestamp: Date.now(),
          solutionQuality: 100 + i,
          computeTime: 1000,
          customersServed: 2
        };

        await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);
      }

      const stats = learningSystem.getStatistics();
      expect(stats.totalEpisodes).toBe(5);
    });
  });

  describe('Traffic Pattern Learning', () => {
    test('should learn traffic patterns from routes', async () => {
      const metrics: LearningMetrics = {
        episodeId: 'ep1',
        timestamp: Date.now(),
        solutionQuality: 100,
        computeTime: 1000,
        customersServed: 2
      };

      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);

      const stats = learningSystem.getStatistics();
      expect(stats.trafficPatternsLearned).toBeGreaterThan(0);
    });

    test('should retrieve traffic predictions', async () => {
      const metrics: LearningMetrics = {
        episodeId: 'ep1',
        timestamp: Date.now(),
        solutionQuality: 100,
        computeTime: 1000,
        customersServed: 2
      };

      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);

      const now = new Date();
      const pattern = learningSystem.getTrafficPrediction(
        'loc1',
        'loc2',
        now.getHours(),
        now.getDay()
      );

      expect(pattern).toBeDefined();
      if (pattern) {
        expect(pattern.avgSpeed).toBeGreaterThan(0);
        expect(pattern.reliability).toBeGreaterThanOrEqual(0);
        expect(pattern.reliability).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Similar Solution Retrieval', () => {
    test('should retrieve similar past solutions', async () => {
      // Add multiple solutions
      for (let i = 0; i < 10; i++) {
        const metrics: LearningMetrics = {
          episodeId: `ep${i}`,
          timestamp: Date.now(),
          solutionQuality: 100 + i * 10,
          computeTime: 1000,
          customersServed: 2
        };

        await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);
      }

      const similar = await learningSystem.retrieveSimilarSolutions(2, 1, 5);

      expect(similar).toBeInstanceOf(Array);
      expect(similar.length).toBeGreaterThan(0);
      expect(similar.length).toBeLessThanOrEqual(5);
    });
  });

  describe('Statistics', () => {
    test('should calculate correct statistics', async () => {
      const metrics1: LearningMetrics = {
        episodeId: 'ep1',
        timestamp: Date.now(),
        solutionQuality: 100,
        computeTime: 1000,
        customersServed: 2
      };

      const metrics2: LearningMetrics = {
        episodeId: 'ep2',
        timestamp: Date.now(),
        solutionQuality: 90,
        computeTime: 1200,
        customersServed: 2
      };

      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics1);
      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics2);

      const stats = learningSystem.getStatistics();

      expect(stats.totalEpisodes).toBe(2);
      expect(stats.avgSolutionQuality).toBe(95);
      expect(stats.avgComputeTime).toBe(1100);
    });

    test('should return zero stats for empty system', () => {
      const stats = learningSystem.getStatistics();

      expect(stats.totalEpisodes).toBe(0);
      expect(stats.avgSolutionQuality).toBe(0);
      expect(stats.avgComputeTime).toBe(0);
    });
  });

  describe('Pattern Export/Import', () => {
    test('should export learned patterns', async () => {
      const metrics: LearningMetrics = {
        episodeId: 'ep1',
        timestamp: Date.now(),
        solutionQuality: 100,
        computeTime: 1000,
        customersServed: 2
      };

      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);

      const exported = learningSystem.exportPatterns();

      expect(exported).toHaveProperty('trafficPatterns');
      expect(exported).toHaveProperty('topSolutions');
      expect(exported).toHaveProperty('statistics');
      expect(exported.trafficPatterns).toBeInstanceOf(Array);
    });

    test('should import patterns', () => {
      const patterns = {
        trafficPatterns: [
          {
            fromLocationId: 'loc1',
            toLocationId: 'loc2',
            timeOfDay: 9,
            dayOfWeek: 1,
            avgSpeed: 45,
            reliability: 0.85
          }
        ]
      };

      learningSystem.importPatterns(patterns);

      const pattern = learningSystem.getTrafficPrediction('loc1', 'loc2', 9, 1);
      expect(pattern).toBeDefined();
      expect(pattern?.avgSpeed).toBe(45);
    });
  });

  describe('Memory Management', () => {
    test('should reset learning state', async () => {
      const metrics: LearningMetrics = {
        episodeId: 'ep1',
        timestamp: Date.now(),
        solutionQuality: 100,
        computeTime: 1000,
        customersServed: 2
      };

      await learningSystem.learnFromSolution(mockSolution, mockCustomers, metrics);

      learningSystem.reset();

      const stats = learningSystem.getStatistics();
      expect(stats.totalEpisodes).toBe(0);
      expect(stats.trafficPatternsLearned).toBe(0);
    });
  });
});
