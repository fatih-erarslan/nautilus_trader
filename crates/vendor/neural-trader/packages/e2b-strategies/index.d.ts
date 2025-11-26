/**
 * @neural-trader/e2b-strategies
 * TypeScript definitions for E2B trading strategies
 */

export interface StrategyConfig {
  apiKey: string;
  secretKey: string;
  baseUrl?: string;
  symbols: string[];
  port?: number;
  host?: string;
  cacheEnabled?: boolean;
  cacheTTL?: number;
  batchWindow?: number;
  circuitBreakerTimeout?: number;
  maxRetries?: number;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
  metricsEnabled?: boolean;
}

export interface MomentumConfig extends StrategyConfig {
  threshold: number;
  positionSize: number;
  interval?: string;
}

export interface NeuralForecastConfig extends StrategyConfig {
  model?: 'lstm' | 'gru' | 'tcn' | 'nbeats' | 'deepar' | 'prophet';
  confidence: number;
  lookbackPeriods?: number;
  forecastHorizon?: number;
  maxPositionSize: number;
  minPositionSize: number;
  useGPU?: boolean;
}

export interface MeanReversionConfig extends StrategyConfig {
  lookbackPeriods?: number;
  entryThreshold: number;
  exitThreshold: number;
  maxPositionSize: number;
}

export interface RiskManagerConfig extends StrategyConfig {
  maxDrawdown: number;
  stopLossPerTrade: number;
  maxPortfolioRisk?: number;
  varConfidence?: number;
  lookbackDays?: number;
}

export interface PortfolioOptimizerConfig extends StrategyConfig {
  method?: 'sharpe' | 'risk_parity' | 'markowitz' | 'black_litterman';
  rebalanceThreshold: number;
  lookbackDays?: number;
  targetReturn?: number;
  riskFreeRate?: number;
}

export interface Trade {
  symbol: string;
  action: 'buy' | 'sell';
  quantity: number;
  price: number;
  orderId: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface Position {
  symbol: string;
  quantity: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPct: number;
}

export interface StrategyStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: number;
  circuitBreakers: Record<string, CircuitBreakerStats>;
  cache?: CacheStats;
  positions: Position[];
}

export interface CircuitBreakerStats {
  closed: boolean;
  failures: number;
  successes: number;
  rejects: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  errors: number;
  size: number;
}

export interface ExecutionResult {
  success: boolean;
  trades?: Trade[];
  errors?: string[];
  summary: {
    total: number;
    success: number;
    failed: number;
    trades: number;
    duration: number;
  };
}

export interface Prediction {
  currentPrice: number;
  predictedPrice: number;
  confidence: number;
  direction: 'up' | 'down';
  upperBound?: number;
  lowerBound?: number;
}

export interface ModelStats {
  trainingSamples: number;
  validationSamples: number;
  loss: number;
  accuracy?: number;
  lastTrained: string;
}

export interface RiskMetrics {
  var: number;
  cvar: number;
  maxDrawdown: number;
  currentDrawdown: number;
  portfolioVolatility: number;
  sharpeRatio: number;
  alerts: RiskAlert[];
}

export interface RiskAlert {
  type: 'STOP_LOSS' | 'MAX_DRAWDOWN' | 'VAR_BREACH' | 'POSITION_LIMIT';
  symbol?: string;
  threshold: number;
  current: number;
  action: string;
  timestamp: string;
}

export interface OptimizationResult {
  allocations: Record<string, number>;
  expectedReturn: number;
  volatility: number;
  sharpe: number;
  stats: {
    iterations: number;
    convergence: boolean;
    duration: number;
  };
}

export declare class BaseStrategy {
  constructor(config: StrategyConfig);
  start(): Promise<void>;
  stop(): Promise<void>;
  restart(): Promise<void>;
  execute(): Promise<ExecutionResult>;
  getStatus(): Promise<StrategyStatus>;
  getPositions(): Promise<Position[]>;
  updateConfig(config: Partial<StrategyConfig>): void;
  getConfig(): StrategyConfig;
  on(event: 'started' | 'stopped' | 'trade' | 'error', handler: Function): void;
  off(event: string, handler: Function): void;
}

export declare class MomentumStrategy extends BaseStrategy {
  constructor(config: MomentumConfig);
  generateSignal(symbol: string): Promise<{ action: 'buy' | 'sell' | 'hold'; momentum: number }>;
}

export declare class NeuralForecastStrategy extends BaseStrategy {
  constructor(config: NeuralForecastConfig);
  trainModel(symbol: string): Promise<void>;
  predict(symbol: string): Promise<Prediction>;
  getModelStats(symbol: string): Promise<ModelStats>;
}

export declare class MeanReversionStrategy extends BaseStrategy {
  constructor(config: MeanReversionConfig);
  calculateZScore(symbol: string): Promise<number>;
}

export declare class RiskManager extends BaseStrategy {
  constructor(config: RiskManagerConfig);
  calculateMetrics(): Promise<RiskMetrics>;
  enforceRiskLimits(): Promise<void>;
  getAlerts(): Promise<RiskAlert[]>;
}

export declare class PortfolioOptimizer extends BaseStrategy {
  constructor(config: PortfolioOptimizerConfig);
  optimize(): Promise<OptimizationResult>;
  rebalance(): Promise<Trade[]>;
  getCurrentAllocations(): Promise<Record<string, number>>;
}

// Testing utilities
export declare namespace testing {
  function mockBroker(): any;
  function mockMarketData(): any;
  function generateBars(symbol: string, count: number, trend?: 'up' | 'down' | 'sideways'): any[];
}

// Re-exports
export * from './momentum';
export * from './neural-forecast';
export * from './mean-reversion';
export * from './risk-manager';
export * from './portfolio-optimizer';
