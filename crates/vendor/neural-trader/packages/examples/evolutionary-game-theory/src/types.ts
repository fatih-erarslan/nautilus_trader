/**
 * Core types for evolutionary game theory
 */

/**
 * Game payoff matrix
 * payoff[strategyA][strategyB] = payoff for player using strategyA against strategyB
 */
export type PayoffMatrix = number[][];

/**
 * Strategy in a game
 */
export interface Strategy {
  id: string;
  name: string;
  description?: string;
  cooperationRate?: number; // For cooperative games
  memory?: number; // Length of history considered
  play: (history: GameHistory, opponentHistory: GameHistory) => number;
}

/**
 * History of moves in an iterated game
 */
export type GameHistory = number[];

/**
 * Game definition
 */
export interface Game {
  id: string;
  name: string;
  description: string;
  payoffMatrix: PayoffMatrix;
  numStrategies: number;
  strategyNames: string[];
  isSymmetric: boolean;
}

/**
 * Population state (strategy frequencies)
 */
export interface PopulationState {
  frequencies: number[];
  generation: number;
  timestamp: number;
  fitnessValues?: number[];
  averageFitness?: number;
}

/**
 * Evolutionarily Stable Strategy result
 */
export interface ESSResult {
  strategy: number[];
  isStable: boolean;
  eigenvalues: number[];
  stabilityMargin: number;
  convergenceTime?: number;
}

/**
 * Tournament participant
 */
export interface TournamentPlayer {
  id: string;
  strategy: Strategy;
  score: number;
  wins: number;
  losses: number;
  draws: number;
  matches: number;
}

/**
 * Tournament result
 */
export interface TournamentResult {
  rankings: TournamentPlayer[];
  totalMatches: number;
  generation: number;
  bestStrategy: Strategy;
  averageScore: number;
  diversityIndex: number;
}

/**
 * Genetic algorithm parameters
 */
export interface GeneticParams {
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  elitismRate: number;
  tournamentSize: number;
  maxGenerations: number;
}

/**
 * Evolution result
 */
export interface EvolutionResult {
  bestStrategy: Strategy;
  bestFitness: number;
  generation: number;
  populationDiversity: number;
  convergenceHistory: number[];
  strategyDistribution: Map<string, number>;
}

/**
 * Swarm configuration
 */
export interface SwarmConfig {
  numAgents: number;
  topology: 'mesh' | 'hierarchical' | 'ring' | 'star';
  communicationRadius?: number;
  learningRate: number;
  explorationRate: number;
}

/**
 * Fitness landscape point
 */
export interface FitnessPoint {
  strategy: number[];
  fitness: number;
  gradient?: number[];
  neighbors?: FitnessPoint[];
}
