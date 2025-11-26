/**
 * Common types for test framework
 */

export interface TestConfig {
  timeout?: number;
  retries?: number;
  verbose?: boolean;
  mockOpenRouter?: boolean;
  mockAgentDB?: boolean;
}

export interface MockOptions {
  delay?: number;
  errorRate?: number;
  responses?: any[];
}

export interface TimeSeriesData {
  timestamps: number[];
  values: number[];
  features?: Record<string, number[]>;
}

export interface MarketData {
  symbol: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TradingSignal {
  timestamp: number;
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  price: number;
  quantity?: number;
}

export interface TestMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  predictions: number;
  errors: number;
}

export interface SwarmMetrics {
  agents: number;
  messages: number;
  convergenceTime: number;
  consensusReached: boolean;
  averageQuality: number;
}
