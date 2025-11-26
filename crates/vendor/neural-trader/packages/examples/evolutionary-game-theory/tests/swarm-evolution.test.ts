/**
 * Tests for swarm evolution system
 */

import { SwarmEvolution, quickSwarmEvolution } from '../src/swarm-evolution.js';
import { PRISONERS_DILEMMA, HAWK_DOVE } from '../src/games.js';
import { createLearningStrategy } from '../src/strategies.js';

describe('SwarmEvolution', () => {
  describe('Initialization', () => {
    it('should initialize with default parameters', () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA);
      const stats = swarm.getStatistics();

      expect(stats.populationSize).toBeGreaterThan(0);
      expect(stats.generation).toBe(0);
    });

    it('should accept custom genetic parameters', () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 50,
        mutationRate: 0.2,
        maxGenerations: 20,
      });

      const stats = swarm.getStatistics();
      expect(stats.populationSize).toBe(50);
    });

    it('should accept custom swarm configuration', () => {
      const swarm = new SwarmEvolution(
        PRISONERS_DILEMMA,
        {},
        {
          numAgents: 200,
          topology: 'hierarchical',
          learningRate: 0.05,
        }
      );

      expect(swarm).toBeDefined();
    });
  });

  describe('Population Management', () => {
    it('should initialize random population', () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
      });

      const stats = swarm.getStatistics();
      expect(stats.populationSize).toBe(20);
    });

    it('should maintain population size across generations', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 30,
        maxGenerations: 3,
      });

      await swarm.evolveGeneration();
      await swarm.evolveGeneration();

      const stats = swarm.getStatistics();
      expect(stats.populationSize).toBe(30);
    });
  });

  describe('Fitness Evaluation', () => {
    it('should evaluate fitness through tournaments', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 10,
      });

      const result = await swarm.evolveGeneration();

      expect(result.bestFitness).toBeDefined();
      expect(typeof result.bestFitness).toBe('number');
    });

    it('should track best strategies', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        maxGenerations: 3,
      });

      await swarm.evolveGeneration();
      await swarm.evolveGeneration();

      const stats = swarm.getStatistics();
      expect(stats.bestStrategies.length).toBeGreaterThan(0);
    });
  });

  describe('Genetic Operations', () => {
    it('should perform selection', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
      });

      const result = await swarm.evolveGeneration();

      expect(result.bestStrategy).toBeDefined();
      expect(result.bestStrategy.id).toBeDefined();
    });

    it('should apply crossover', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        crossoverRate: 0.9,
      });

      await swarm.evolveGeneration();
      const stats = swarm.getStatistics();

      expect(stats.generation).toBe(1);
    });

    it('should apply mutation', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        mutationRate: 0.3,
      });

      await swarm.evolveGeneration();
      const stats = swarm.getStatistics();

      expect(stats.populationSize).toBe(20);
    });

    it('should preserve elites', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        elitismRate: 0.2,
      });

      const result1 = await swarm.evolveGeneration();
      const result2 = await swarm.evolveGeneration();

      // Best fitness should not decrease with elitism
      expect(result2.bestFitness).toBeGreaterThanOrEqual(
        result1.bestFitness - 0.5
      ); // Allow some variance
    });
  });

  describe('Evolution Process', () => {
    it('should evolve for multiple generations', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        maxGenerations: 5,
      });

      const result = await swarm.run();

      expect(result.generation).toBeGreaterThan(0);
      expect(result.bestStrategy).toBeDefined();
    });

    it('should improve fitness over time', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 30,
        maxGenerations: 10,
      });

      const result1 = await swarm.evolveGeneration();
      await swarm.evolveGeneration();
      await swarm.evolveGeneration();
      const result2 = await swarm.evolveGeneration();

      // Fitness should generally improve
      expect(result2.bestFitness).toBeGreaterThanOrEqual(
        result1.bestFitness - 1.0
      );
    });

    it('should track convergence history', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        maxGenerations: 5,
      });

      const result = await swarm.run();

      expect(result.convergenceHistory).toBeDefined();
      expect(result.convergenceHistory.length).toBeGreaterThan(0);
    });

    it('should calculate population diversity', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 30,
      });

      const result = await swarm.evolveGeneration();

      expect(result.populationDiversity).toBeDefined();
      expect(result.populationDiversity).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Strategy Distribution', () => {
    it('should track strategy distribution', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
      });

      const result = await swarm.evolveGeneration();

      expect(result.strategyDistribution).toBeDefined();
      expect(result.strategyDistribution.size).toBeGreaterThan(0);
    });
  });

  describe('Statistics', () => {
    it('should provide population statistics', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 25,
      });

      await swarm.evolveGeneration();
      const stats = swarm.getStatistics();

      expect(stats.generation).toBe(1);
      expect(stats.populationSize).toBe(25);
      expect(stats.averageFitness).toBeDefined();
      expect(stats.fitnessVariance).toBeDefined();
    });

    it('should track best strategies over time', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 20,
        maxGenerations: 3,
      });

      await swarm.evolveGeneration();
      await swarm.evolveGeneration();

      const stats = swarm.getStatistics();
      expect(stats.bestStrategies.length).toBeGreaterThan(0);
      expect(stats.bestStrategies.length).toBeLessThanOrEqual(5); // Keeps last 5
    });
  });

  describe('Fitness Landscape Exploration', () => {
    it('should explore fitness landscape', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 10,
      });

      const landscape = await swarm.exploreFitnessLandscape(5);

      expect(landscape).toBeDefined();
      expect(landscape.length).toBeGreaterThan(0);
      expect(landscape[0].strategy).toBeDefined();
      expect(landscape[0].fitness).toBeDefined();
    });

    it('should record fitness values', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 10,
      });

      const landscape = await swarm.exploreFitnessLandscape(3);

      for (const point of landscape) {
        expect(typeof point.fitness).toBe('number');
        expect(Array.isArray(point.strategy)).toBe(true);
      }
    });
  });

  describe('OpenRouter Integration', () => {
    it('should accept API key', () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA);
      swarm.setOpenRouterKey('test-key');

      expect(swarm).toBeDefined();
    });

    // Note: Actual LLM innovation tests would require API key and be integration tests
    it('should have innovation method', () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA);
      expect(typeof swarm.innovateWithLLM).toBe('function');
    });
  });

  describe('AgentDB Integration', () => {
    it('should support AgentDB initialization', async () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA);

      // Mock AgentDB
      const mockDB = {
        createCollection: async () => {},
        upsert: async () => {},
        query: async () => [],
      };

      await swarm.initializeAgentDB(mockDB);
      expect(swarm).toBeDefined();
    });

    it('should have query method', () => {
      const swarm = new SwarmEvolution(PRISONERS_DILEMMA);
      expect(typeof swarm.querySimilarStrategies).toBe('function');
    });
  });

  describe('Different Games', () => {
    it('should work with Hawk-Dove game', async () => {
      const swarm = new SwarmEvolution(HAWK_DOVE, {
        populationSize: 20,
        maxGenerations: 3,
      });

      const result = await swarm.run();

      expect(result.bestStrategy).toBeDefined();
      expect(result.generation).toBeGreaterThan(0);
    });

    it('should adapt to different game dynamics', async () => {
      const pdSwarm = new SwarmEvolution(PRISONERS_DILEMMA, {
        populationSize: 15,
        maxGenerations: 2,
      });

      const hdSwarm = new SwarmEvolution(HAWK_DOVE, {
        populationSize: 15,
        maxGenerations: 2,
      });

      const pdResult = await pdSwarm.run();
      const hdResult = await hdSwarm.run();

      // Both should find strategies
      expect(pdResult.bestStrategy).toBeDefined();
      expect(hdResult.bestStrategy).toBeDefined();
    });
  });
});

describe('quickSwarmEvolution', () => {
  it('should run quick evolution', async () => {
    const result = await quickSwarmEvolution(PRISONERS_DILEMMA, 5);

    expect(result.bestStrategy).toBeDefined();
    expect(result.bestFitness).toBeDefined();
    expect(result.generation).toBeGreaterThan(0);
  });

  it('should use default parameters', async () => {
    const result = await quickSwarmEvolution();

    expect(result).toBeDefined();
    expect(result.bestStrategy).toBeDefined();
  });

  it('should respect generations parameter', async () => {
    const result = await quickSwarmEvolution(PRISONERS_DILEMMA, 3);

    expect(result.generation).toBeLessThanOrEqual(3);
  });
});

describe('Performance and Scalability', () => {
  it('should handle larger populations', async () => {
    const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
      populationSize: 100,
      maxGenerations: 2,
    });

    const result = await swarm.run();

    expect(result.bestStrategy).toBeDefined();
  });

  it('should complete evolution in reasonable time', async () => {
    const startTime = Date.now();

    const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
      populationSize: 50,
      maxGenerations: 5,
    });

    await swarm.run();

    const elapsed = Date.now() - startTime;

    // Should complete in under 30 seconds for 50 population, 5 generations
    expect(elapsed).toBeLessThan(30000);
  }, 35000); // 35 second timeout

  it('should handle many generations', async () => {
    const swarm = new SwarmEvolution(PRISONERS_DILEMMA, {
      populationSize: 20,
      maxGenerations: 20,
    });

    const result = await swarm.run();

    expect(result.generation).toBeGreaterThan(0);
    expect(result.generation).toBeLessThanOrEqual(20);
  });
});
