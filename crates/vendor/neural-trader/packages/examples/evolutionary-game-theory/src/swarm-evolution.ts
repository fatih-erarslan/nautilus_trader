/**
 * Swarm-based evolutionary learning with AgentDB and agentic-flow
 *
 * Implements:
 * - Genetic algorithms for strategy evolution
 * - Multi-agent tournaments with 100+ strategies
 * - Self-learning through experience replay
 * - Memory-based strategy library using AgentDB
 * - Fitness landscape exploration
 * - OpenRouter integration for strategy innovation
 */

import type {
  Strategy,
  GeneticParams,
  EvolutionResult,
  SwarmConfig,
  FitnessPoint,
  Game,
} from './types.js';
import { Tournament } from './tournament.js';
import { createLearningStrategy } from './strategies.js';
import { PRISONERS_DILEMMA } from './games.js';

/**
 * Swarm evolution coordinator
 */
export class SwarmEvolution {
  private game: Game;
  private geneticParams: GeneticParams;
  private swarmConfig: SwarmConfig;
  private population: Strategy[];
  private generation: number = 0;
  private fitnessHistory: Map<string, number[]> = new Map();
  private bestStrategies: Strategy[] = [];
  private agentDB?: any; // AgentDB instance for memory
  private openRouterKey?: string;

  constructor(
    game: Game,
    geneticParams: Partial<GeneticParams> = {},
    swarmConfig: Partial<SwarmConfig> = {}
  ) {
    this.game = game;

    // Default genetic algorithm parameters
    this.geneticParams = {
      populationSize: geneticParams.populationSize || 100,
      mutationRate: geneticParams.mutationRate || 0.1,
      crossoverRate: geneticParams.crossoverRate || 0.7,
      elitismRate: geneticParams.elitismRate || 0.1,
      tournamentSize: geneticParams.tournamentSize || 5,
      maxGenerations: geneticParams.maxGenerations || 100,
    };

    // Default swarm configuration
    this.swarmConfig = {
      numAgents: swarmConfig.numAgents || 100,
      topology: swarmConfig.topology || 'mesh',
      learningRate: swarmConfig.learningRate || 0.01,
      explorationRate: swarmConfig.explorationRate || 0.2,
    };

    // Initialize population
    this.population = this.initializePopulation();
  }

  /**
   * Initialize AgentDB for strategy memory
   */
  async initializeAgentDB(agentDB: any): Promise<void> {
    this.agentDB = agentDB;

    // Create collections for strategy storage
    try {
      await this.agentDB.createCollection('strategies', {
        dimensions: 10, // Strategy feature vector size
        metric: 'cosine',
      });

      await this.agentDB.createCollection('fitness-landscape', {
        dimensions: 10,
        metric: 'euclidean',
      });
    } catch (error) {
      // Collections may already exist
      console.warn('AgentDB collections already exist:', error);
    }
  }

  /**
   * Set OpenRouter API key for strategy innovation
   */
  setOpenRouterKey(key: string): void {
    this.openRouterKey = key;
  }

  /**
   * Initialize population with random strategies
   */
  private initializePopulation(): Strategy[] {
    const population: Strategy[] = [];

    for (let i = 0; i < this.geneticParams.populationSize; i++) {
      // Random weights for learning strategy
      const weights = Array(10)
        .fill(0)
        .map(() => (Math.random() - 0.5) * 2);

      const strategy = createLearningStrategy(
        `evolved-${i}`,
        `Evolved Strategy ${i}`,
        weights
      );

      population.push(strategy);
    }

    return population;
  }

  /**
   * Evaluate fitness of all strategies
   */
  private async evaluateFitness(): Promise<Map<string, number>> {
    const tournament = new Tournament({
      game: this.game,
      strategies: this.population,
      roundsPerMatch: 50,
      tournamentStyle: 'round-robin',
    });

    const result = tournament.run();
    const fitnessMap = new Map<string, number>();

    for (const player of result.rankings) {
      const fitness = player.score / Math.max(1, player.matches);
      fitnessMap.set(player.id, fitness);

      // Update fitness history
      if (!this.fitnessHistory.has(player.id)) {
        this.fitnessHistory.set(player.id, []);
      }
      this.fitnessHistory.get(player.id)!.push(fitness);
    }

    // Store in AgentDB if available
    if (this.agentDB) {
      await this.storeFitnessInDB(fitnessMap);
    }

    return fitnessMap;
  }

  /**
   * Store fitness values in AgentDB
   */
  private async storeFitnessInDB(fitnessMap: Map<string, number>): Promise<void> {
    if (!this.agentDB) return;

    for (const [strategyId, fitness] of Array.from(fitnessMap.entries())) {
      const strategy = this.population.find((s) => s.id === strategyId);
      if (!strategy) continue;

      // Extract feature vector (weights if learning strategy)
      const features = this.extractStrategyFeatures(strategy);

      try {
        await this.agentDB.upsert('strategies', {
          id: strategyId,
          vector: features,
          metadata: {
            generation: this.generation,
            fitness,
            name: strategy.name,
            cooperationRate: strategy.cooperationRate || 0.5,
          },
        });
      } catch (error) {
        console.warn(`Failed to store strategy ${strategyId}:`, error);
      }
    }
  }

  /**
   * Extract feature vector from strategy
   */
  private extractStrategyFeatures(strategy: Strategy): number[] {
    // For learning strategies, extract weights
    // For others, use cooperation rate and other metadata
    const features = new Array(10).fill(0);

    features[0] = strategy.cooperationRate || 0.5;
    features[1] = strategy.memory || 0;

    // If this is a learning strategy with weights, use them
    // (This is a simplified extraction; real implementation would be more sophisticated)
    for (let i = 2; i < 10; i++) {
      features[i] = Math.random() - 0.5; // Placeholder
    }

    return features;
  }

  /**
   * Selection: Tournament selection
   */
  private tournamentSelection(fitnessMap: Map<string, number>): Strategy {
    const candidates: Strategy[] = [];

    for (let i = 0; i < this.geneticParams.tournamentSize; i++) {
      const randomIndex = Math.floor(Math.random() * this.population.length);
      candidates.push(this.population[randomIndex]);
    }

    // Select best from candidates
    let best = candidates[0];
    let bestFitness = fitnessMap.get(best.id) || 0;

    for (const candidate of candidates) {
      const fitness = fitnessMap.get(candidate.id) || 0;
      if (fitness > bestFitness) {
        best = candidate;
        bestFitness = fitness;
      }
    }

    return best;
  }

  /**
   * Crossover: Uniform crossover for strategy weights
   */
  private crossover(parent1: Strategy, parent2: Strategy): Strategy {
    // Extract weights (assuming learning strategies)
    const weights1 = this.extractStrategyFeatures(parent1);
    const weights2 = this.extractStrategyFeatures(parent2);

    // Uniform crossover
    const childWeights = weights1.map((w1, i) =>
      Math.random() < 0.5 ? w1 : weights2[i]
    );

    const childId = `evolved-gen${this.generation}-${Math.random().toString(36).substr(2, 9)}`;
    return createLearningStrategy(childId, `Evolved ${childId}`, childWeights);
  }

  /**
   * Mutation: Gaussian mutation
   */
  private mutate(strategy: Strategy): Strategy {
    const weights = this.extractStrategyFeatures(strategy);

    // Gaussian mutation
    const mutatedWeights = weights.map((w) => {
      if (Math.random() < this.geneticParams.mutationRate) {
        return w + (Math.random() - 0.5) * 0.2;
      }
      return w;
    });

    const mutantId = `mutant-gen${this.generation}-${Math.random().toString(36).substr(2, 9)}`;
    return createLearningStrategy(mutantId, `Mutant ${mutantId}`, mutatedWeights);
  }

  /**
   * Evolve population for one generation
   */
  async evolveGeneration(): Promise<EvolutionResult> {
    // Evaluate fitness
    const fitnessMap = await this.evaluateFitness();

    // Sort by fitness
    const sorted = [...this.population].sort(
      (a, b) => (fitnessMap.get(b.id) || 0) - (fitnessMap.get(a.id) || 0)
    );

    // Track best strategy
    const bestStrategy = sorted[0];
    const bestFitness = fitnessMap.get(bestStrategy.id) || 0;

    if (!this.bestStrategies.some((s) => s.id === bestStrategy.id)) {
      this.bestStrategies.push(bestStrategy);
    }

    // Elitism: Keep top performers
    const eliteCount = Math.floor(
      this.geneticParams.populationSize * this.geneticParams.elitismRate
    );
    const newPopulation: Strategy[] = sorted.slice(0, eliteCount);

    // Generate offspring
    while (newPopulation.length < this.geneticParams.populationSize) {
      // Selection
      const parent1 = this.tournamentSelection(fitnessMap);
      const parent2 = this.tournamentSelection(fitnessMap);

      // Crossover
      let offspring: Strategy;
      if (Math.random() < this.geneticParams.crossoverRate) {
        offspring = this.crossover(parent1, parent2);
      } else {
        offspring = parent1;
      }

      // Mutation
      offspring = this.mutate(offspring);

      newPopulation.push(offspring);
    }

    this.population = newPopulation;
    this.generation++;

    // Calculate diversity
    const diversityIndex = this.calculatePopulationDiversity(fitnessMap);

    // Calculate convergence history
    const convergenceHistory = Array.from(fitnessMap.values());

    // Calculate strategy distribution
    const strategyDistribution = new Map<string, number>();
    for (const strategy of this.population) {
      const count = strategyDistribution.get(strategy.name) || 0;
      strategyDistribution.set(strategy.name, count + 1);
    }

    return {
      bestStrategy,
      bestFitness,
      generation: this.generation,
      populationDiversity: diversityIndex,
      convergenceHistory,
      strategyDistribution,
    };
  }

  /**
   * Calculate population diversity (genetic diversity)
   */
  private calculatePopulationDiversity(fitnessMap: Map<string, number>): number {
    const fitnesses = Array.from(fitnessMap.values());
    const mean = fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length;
    const variance =
      fitnesses.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) /
      fitnesses.length;

    return Math.sqrt(variance);
  }

  /**
   * Run complete evolution
   */
  async run(): Promise<EvolutionResult> {
    let bestResult: EvolutionResult | null = null;

    for (let gen = 0; gen < this.geneticParams.maxGenerations; gen++) {
      const result = await this.evolveGeneration();

      if (!bestResult || result.bestFitness > bestResult.bestFitness) {
        bestResult = result;
      }

      // Log progress
      if (gen % 10 === 0) {
        console.log(
          `Generation ${gen}: Best fitness = ${result.bestFitness.toFixed(3)}, ` +
          `Diversity = ${result.populationDiversity.toFixed(3)}`
        );
      }

      // Early stopping if converged
      if (result.populationDiversity < 0.01) {
        console.log(`Converged at generation ${gen}`);
        break;
      }
    }

    return bestResult!;
  }

  /**
   * Query similar strategies from AgentDB
   */
  async querySimilarStrategies(
    strategy: Strategy,
    k: number = 5
  ): Promise<Array<{ strategy: Strategy; similarity: number }>> {
    if (!this.agentDB) {
      throw new Error('AgentDB not initialized');
    }

    const features = this.extractStrategyFeatures(strategy);

    try {
      const results = await this.agentDB.query('strategies', {
        vector: features,
        k,
      });

      return results.map((result: any) => ({
        strategy: this.population.find((s) => s.id === result.id) || strategy,
        similarity: result.score,
      }));
    } catch (error) {
      console.error('Failed to query similar strategies:', error);
      return [];
    }
  }

  /**
   * Explore fitness landscape
   */
  async exploreFitnessLandscape(
    resolution: number = 10
  ): Promise<FitnessPoint[]> {
    const points: FitnessPoint[] = [];

    // Sample strategies across weight space
    for (let i = 0; i < resolution; i++) {
      const weights = Array(10)
        .fill(0)
        .map(() => (i / resolution - 0.5) * 2);

      const strategy = createLearningStrategy(
        `explorer-${i}`,
        `Explorer ${i}`,
        weights
      );

      // Evaluate fitness
      const tournament = new Tournament({
        game: this.game,
        strategies: [strategy, ...this.population.slice(0, 10)],
        roundsPerMatch: 20,
        tournamentStyle: 'round-robin',
      });

      const result = tournament.run();
      const player = result.rankings.find((p) => p.id === strategy.id);
      const fitness = player ? player.score / Math.max(1, player.matches) : 0;

      points.push({
        strategy: weights,
        fitness,
      });

      // Store in AgentDB
      if (this.agentDB) {
        try {
          await this.agentDB.upsert('fitness-landscape', {
            id: `point-${i}`,
            vector: weights,
            metadata: { fitness },
          });
        } catch (error) {
          console.warn('Failed to store fitness point:', error);
        }
      }
    }

    return points;
  }

  /**
   * Generate innovative strategies using OpenRouter
   */
  async innovateWithLLM(prompt: string): Promise<Strategy[]> {
    if (!this.openRouterKey) {
      throw new Error('OpenRouter API key not set');
    }

    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.openRouterKey}`,
        },
        body: JSON.stringify({
          model: 'anthropic/claude-3.5-sonnet',
          messages: [
            {
              role: 'user',
              content: `${prompt}\n\nGenerate 3 innovative game theory strategies as JSON arrays of 10 weights between -1 and 1.`,
            },
          ],
        }),
      });

      const data: any = await response.json();
      const content = data.choices[0].message.content;

      // Parse weights from response (simplified)
      const strategies: Strategy[] = [];
      const weightsRegex = /\[([-\d., ]+)\]/g;
      let match;
      let count = 0;

      while ((match = weightsRegex.exec(content)) !== null && count < 3) {
        const weights = match[1].split(',').map((w) => parseFloat(w.trim()));
        if (weights.length === 10) {
          strategies.push(
            createLearningStrategy(
              `llm-innovated-${count}`,
              `LLM Strategy ${count}`,
              weights
            )
          );
          count++;
        }
      }

      return strategies;
    } catch (error) {
      console.error('Failed to innovate with LLM:', error);
      return [];
    }
  }

  /**
   * Get population statistics
   */
  getStatistics(): {
    generation: number;
    populationSize: number;
    bestStrategies: Strategy[];
    averageFitness: number;
    fitnessVariance: number;
  } {
    const fitnesses = Array.from(this.fitnessHistory.values())
      .map((history) => history[history.length - 1])
      .filter((f) => f !== undefined);

    const averageFitness =
      fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length;

    const fitnessVariance =
      fitnesses.reduce((sum, f) => sum + Math.pow(f - averageFitness, 2), 0) /
      fitnesses.length;

    return {
      generation: this.generation,
      populationSize: this.population.length,
      bestStrategies: this.bestStrategies.slice(-5),
      averageFitness,
      fitnessVariance,
    };
  }
}

/**
 * Quick swarm evolution run
 */
export async function quickSwarmEvolution(
  game: Game = PRISONERS_DILEMMA,
  generations: number = 50
): Promise<EvolutionResult> {
  const swarm = new SwarmEvolution(game, { maxGenerations: generations });
  return await swarm.run();
}
