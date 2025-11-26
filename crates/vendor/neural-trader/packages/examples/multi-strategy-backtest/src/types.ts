/**
 * Core types for multi-strategy backtesting system
 */

export interface MarketData {
  timestamp: number;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  entryTime: number;
  side: 'long' | 'short';
  strategy: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: number;
  strategy: string;
  commission: number;
  slippage: number;
}

export interface StrategySignal {
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  strength: number; // 0-1
  confidence: number; // 0-1
  strategy: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface StrategyPerformance {
  strategyName: string;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  trades: number;
  avgWin: number;
  avgLoss: number;
  calmarRatio: number;
  sortinoRatio: number;
}

export interface BacktestConfig {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  symbols: string[];
  strategies: StrategyConfig[];
  commission: number;
  slippage: number;
  walkForwardPeriods?: number;
  rebalanceFrequency?: 'daily' | 'weekly' | 'monthly';
}

export interface StrategyConfig {
  name: string;
  type: 'momentum' | 'mean-reversion' | 'pairs-trading' | 'market-making';
  initialWeight: number;
  parameters: Record<string, any>;
  enabled: boolean;
}

export interface RegimeDetection {
  regime: 'bull' | 'bear' | 'sideways' | 'high-volatility' | 'low-volatility';
  confidence: number;
  timestamp: number;
  indicators: {
    trend: number;
    volatility: number;
    correlation: number;
  };
}

export interface PortfolioState {
  timestamp: number;
  equity: number;
  cash: number;
  positions: Position[];
  strategyWeights: Record<string, number>;
  regime: RegimeDetection;
}

export interface LearningState {
  episodeCount: number;
  totalReward: number;
  explorationRate: number;
  qTable: Map<string, Map<string, number>>;
  experienceBuffer: Experience[];
  bestPerformance: StrategyPerformance[];
}

export interface Experience {
  state: string;
  action: string;
  reward: number;
  nextState: string;
  done: boolean;
  timestamp: number;
}

export interface SwarmParticle {
  id: string;
  position: Record<string, number>;
  velocity: Record<string, number>;
  bestPosition: Record<string, number>;
  bestScore: number;
  currentScore: number;
}

export interface SwarmConfig {
  particleCount: number;
  maxIterations: number;
  inertia: number;
  cognitiveWeight: number;
  socialWeight: number;
  bounds: Record<string, [number, number]>;
}

export interface OptimizationResult {
  bestParameters: Record<string, number>;
  bestScore: number;
  convergenceHistory: number[];
  evaluations: number;
  timeElapsed: number;
}

export interface Strategy {
  name: string;
  type: string;
  generateSignal(data: MarketData[], parameters: Record<string, any>): StrategySignal;
  updateParameters(parameters: Record<string, any>): void;
  getParameters(): Record<string, any>;
}
