/**
 * Type definitions for dynamic pricing system
 */

export interface MarketContext {
  timestamp: number;
  dayOfWeek: number;
  hour: number;
  isHoliday: boolean;
  isPromotion: boolean;
  seasonality: number;
  competitorPrices: number[];
  inventory: number;
  demand: number;
}

export interface CustomerSegment {
  id: string;
  priceElasticity: number;
  valuePerception: number;
  competitorAwareness: number;
  purchaseHistory: Purchase[];
  sensitivity: number;
}

export interface Purchase {
  price: number;
  timestamp: number;
  quantity: number;
  alternatives: number[];
  converted: boolean;
}

export interface PricingStrategy {
  name: string;
  calculate: (context: MarketContext, basePrice: number) => number;
  features?: string[];
}

export interface RLState {
  price: number;
  demand: number;
  inventory: number;
  competitorAvgPrice: number;
  timeFeatures: number[];
  normalized: number[];
}

export interface RLAction {
  priceMultiplier: number; // e.g., 0.8 to 1.2 (80% to 120% of base)
  index: number;
}

export interface RLExperience {
  state: RLState;
  action: RLAction;
  reward: number;
  nextState: RLState;
  done: boolean;
}

export interface ElasticityEstimate {
  mean: number;
  std: number;
  confidence: number;
  samples: number;
}

export interface PriceRecommendation {
  price: number;
  strategy: string;
  expectedRevenue: number;
  expectedDemand: number;
  confidence: number;
  uncertaintyBounds: [number, number];
  competitivePosition: string;
}

export interface SwarmAgent {
  id: string;
  strategy: PricingStrategy;
  performance: AgentPerformance;
  memory: any[];
}

export interface AgentPerformance {
  totalRevenue: number;
  avgDemand: number;
  priceCompetitiveness: number;
  customerSatisfaction: number;
  inventoryTurnover: number;
}

export interface BanditArm {
  pricePoint: number;
  pulls: number;
  totalReward: number;
  avgReward: number;
  ucb: number;
}

export interface ConformalPrediction {
  point: number;
  lower: number;
  upper: number;
  coverage: number;
}

export type RLAlgorithm = 'q-learning' | 'dqn' | 'ppo' | 'sarsa' | 'actor-critic';

export interface RLConfig {
  algorithm: RLAlgorithm;
  learningRate: number;
  discountFactor: number;
  epsilon: number;
  epsilonDecay: number;
  minEpsilon: number;
  batchSize?: number;
  memorySize?: number;
  targetUpdateFreq?: number;
}

export interface SwarmConfig {
  numAgents: number;
  strategies: string[];
  communicationTopology: 'mesh' | 'star' | 'ring';
  consensusMechanism: 'voting' | 'weighted' | 'tournament';
  explorationRate: number;
}

export interface CompetitorAnalysis {
  prices: number[];
  avgPrice: number;
  minPrice: number;
  maxPrice: number;
  priceDispersion: number;
  marketPosition: 'leader' | 'follower' | 'aggressive' | 'premium';
  recommendedPosition: string;
}
