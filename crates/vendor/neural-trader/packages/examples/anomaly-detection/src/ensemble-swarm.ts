import type { AnomalyPoint } from './index';
import { IsolationForest } from './algorithms/isolation-forest';
import { LSTMAutoencoder } from './algorithms/lstm-autoencoder';
import { VAE } from './algorithms/vae';
import { OneClassSVM } from './algorithms/one-class-svm';

export interface SwarmConfig {
  featureDimensions: number;
  populationSize: number;
  maxGenerations: number;
  crossoverRate: number;
  mutationRate: number;
}

export interface AlgorithmScore {
  algorithm: string;
  score: number;
  weight: number;
}

interface Individual {
  weights: number[]; // Weights for each algorithm
  fitness: number;
}

/**
 * Swarm-based ensemble coordinator for anomaly detection algorithms
 *
 * Uses genetic algorithm to optimize algorithm weights based on performance
 */
export class EnsembleSwarm {
  private algorithms: Map<string, any>;
  private population: Individual[] = [];
  private bestIndividual?: Individual;
  private generation = 0;

  constructor(private config: SwarmConfig) {
    this.algorithms = new Map([
      ['isolation-forest', new IsolationForest(config.featureDimensions)],
      ['lstm-ae', new LSTMAutoencoder(config.featureDimensions)],
      ['vae', new VAE(config.featureDimensions)],
      ['one-class-svm', new OneClassSVM(config.featureDimensions)],
    ]);

    this.initializePopulation();
  }

  /**
   * Initialize random population of weight combinations
   */
  private initializePopulation(): void {
    const numAlgorithms = this.algorithms.size;

    for (let i = 0; i < this.config.populationSize; i++) {
      const weights = Array.from({ length: numAlgorithms }, () => Math.random());
      const sum = weights.reduce((a, b) => a + b, 0);
      const normalizedWeights = weights.map(w => w / sum); // Normalize to sum to 1

      this.population.push({
        weights: normalizedWeights,
        fitness: 0,
      });
    }
  }

  /**
   * Train ensemble on labeled data
   */
  async train(trainingData: AnomalyPoint[]): Promise<void> {
    console.log(`Training ensemble with ${trainingData.length} samples...`);

    // Train individual algorithms
    const algorithmPromises = Array.from(this.algorithms.entries()).map(
      async ([name, algorithm]) => {
        try {
          await algorithm.train(trainingData);
          console.log(`${name} training complete`);
        } catch (error) {
          console.error(`${name} training failed:`, error);
        }
      }
    );

    await Promise.all(algorithmPromises);

    // Evolve ensemble weights
    for (let gen = 0; gen < this.config.maxGenerations; gen++) {
      await this.evolveGeneration(trainingData);
      this.generation = gen;

      if (gen % 10 === 0) {
        console.log(
          `Generation ${gen}: Best fitness = ${this.bestIndividual?.fitness.toFixed(4)}`
        );
      }
    }

    console.log('Ensemble training complete');
    console.log('Best weights:', this.bestIndividual?.weights);
  }

  /**
   * Evolve population for one generation
   */
  private async evolveGeneration(validationData: AnomalyPoint[]): Promise<void> {
    // Evaluate fitness
    await this.evaluateFitness(validationData);

    // Sort by fitness (descending)
    this.population.sort((a, b) => b.fitness - a.fitness);
    this.bestIndividual = this.population[0];

    // Selection: keep top 50%
    const survivors = this.population.slice(0, Math.floor(this.population.length / 2));

    // Generate offspring
    const offspring: Individual[] = [];
    while (offspring.length < this.config.populationSize - survivors.length) {
      // Tournament selection
      const parent1 = this.tournamentSelect(survivors);
      const parent2 = this.tournamentSelect(survivors);

      // Crossover
      if (Math.random() < this.config.crossoverRate) {
        const child = this.crossover(parent1, parent2);
        offspring.push(child);
      } else {
        offspring.push({ ...parent1, fitness: 0 });
      }
    }

    // Mutation
    offspring.forEach(individual => {
      if (Math.random() < this.config.mutationRate) {
        this.mutate(individual);
      }
    });

    // New population
    this.population = [...survivors, ...offspring];
  }

  /**
   * Evaluate fitness of all individuals
   */
  private async evaluateFitness(validationData: AnomalyPoint[]): Promise<void> {
    for (const individual of this.population) {
      let truePositives = 0;
      let falsePositives = 0;
      let trueNegatives = 0;
      let falseNegatives = 0;

      for (const point of validationData) {
        const prediction = this.predictWithWeights(point, individual.weights);
        const isAnomaly = prediction > 0.5;
        const actualAnomaly = point.label === 'anomaly';

        if (isAnomaly && actualAnomaly) truePositives++;
        else if (isAnomaly && !actualAnomaly) falsePositives++;
        else if (!isAnomaly && !actualAnomaly) trueNegatives++;
        else falseNegatives++;
      }

      // Fitness = F1 score
      const precision = truePositives / (truePositives + falsePositives || 1);
      const recall = truePositives / (truePositives + falseNegatives || 1);
      const f1 = 2 * (precision * recall) / (precision + recall || 1);

      individual.fitness = f1;
    }
  }

  /**
   * Tournament selection
   */
  private tournamentSelect(population: Individual[]): Individual {
    const tournamentSize = 3;
    const tournament = Array.from(
      { length: tournamentSize },
      () => population[Math.floor(Math.random() * population.length)]
    );
    return tournament.reduce((best, current) =>
      current.fitness > best.fitness ? current : best
    );
  }

  /**
   * Crossover two parents
   */
  private crossover(parent1: Individual, parent2: Individual): Individual {
    const crossoverPoint = Math.floor(Math.random() * parent1.weights.length);
    const childWeights = [
      ...parent1.weights.slice(0, crossoverPoint),
      ...parent2.weights.slice(crossoverPoint),
    ];

    // Re-normalize
    const sum = childWeights.reduce((a, b) => a + b, 0);
    const normalizedWeights = childWeights.map(w => w / sum);

    return {
      weights: normalizedWeights,
      fitness: 0,
    };
  }

  /**
   * Mutate an individual
   */
  private mutate(individual: Individual): void {
    const mutationStrength = 0.1;
    const mutationIndex = Math.floor(Math.random() * individual.weights.length);

    individual.weights[mutationIndex] += (Math.random() - 0.5) * mutationStrength;
    individual.weights[mutationIndex] = Math.max(0, individual.weights[mutationIndex]);

    // Re-normalize
    const sum = individual.weights.reduce((a, b) => a + b, 0);
    individual.weights = individual.weights.map(w => w / sum);
  }

  /**
   * Predict using specific weights
   */
  private predictWithWeights(point: AnomalyPoint, weights: number[]): number {
    const algorithmArray = Array.from(this.algorithms.values());
    let weightedScore = 0;

    algorithmArray.forEach((algorithm, idx) => {
      const score = algorithm.predict(point);
      weightedScore += score * weights[idx];
    });

    return weightedScore;
  }

  /**
   * Predict using best ensemble
   */
  predictEnsemble(point: AnomalyPoint): number {
    if (!this.bestIndividual) {
      throw new Error('Ensemble not trained yet');
    }
    return this.predictWithWeights(point, this.bestIndividual.weights);
  }

  /**
   * Get scores from all algorithms
   */
  getAlgorithmScores(point: AnomalyPoint): Record<string, number> {
    const scores: Record<string, number> = {};

    for (const [name, algorithm] of this.algorithms) {
      scores[name] = algorithm.predict(point);
    }

    return scores;
  }

  /**
   * Get algorithm weights
   */
  getAlgorithmWeights(): AlgorithmScore[] {
    if (!this.bestIndividual) {
      return [];
    }

    const names = Array.from(this.algorithms.keys());
    return names.map((name, idx) => ({
      algorithm: name,
      score: 0,
      weight: this.bestIndividual!.weights[idx],
    }));
  }

  /**
   * Get current generation
   */
  getGeneration(): number {
    return this.generation;
  }

  /**
   * Get best fitness
   */
  getBestFitness(): number {
    return this.bestIndividual?.fitness ?? 0;
  }
}
