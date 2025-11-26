/**
 * @neural-trader/core
 *
 * Core TypeScript types and interfaces for Neural Trader.
 * This package contains ONLY type definitions with zero runtime dependencies.
 *
 * Use this as the foundation for all Neural Trader packages.
 */

// ============================================================================
// Broker Types
// ============================================================================

/** Broker configuration for connection */
export interface BrokerConfig {
  brokerType: string;
  apiKey: string;
  apiSecret: string;
  baseUrl?: string;
  paperTrading: boolean;
  exchange?: string;
}

/** Order placement request */
export interface OrderRequest {
  symbol: string;
  side: string;
  orderType: string;
  quantity: number;
  limitPrice?: number;
  stopPrice?: number;
  timeInForce: string;
}

/** Order response from broker */
export interface OrderResponse {
  orderId: string;
  brokerOrderId: string;
  status: string;
  filledQuantity: number;
  filledPrice?: number;
  timestamp: string;
}

/** Account balance information */
export interface AccountBalance {
  cash: number;
  equity: number;
  buyingPower: number;
  currency: string;
}

// ============================================================================
// Neural Model Types
// ============================================================================

/** Model type enumeration */
export enum ModelType {
  NHITS = 'NHITS',
  LSTMAttention = 'LSTMAttention',
  Transformer = 'Transformer',
}

/** Model configuration */
export interface ModelConfig {
  modelType: string;
  inputSize: number;
  horizon: number;
  hiddenSize: number;
  numLayers: number;
  dropout: number;
  learningRate: number;
}

/** Training configuration */
export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  earlyStoppingPatience: number;
  useGpu: boolean;
}

/** Training metrics */
export interface TrainingMetrics {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainMae: number;
  valMae: number;
}

/** Prediction result with confidence intervals */
export interface PredictionResult {
  predictions: number[];
  lowerBound: number[];
  upperBound: number[];
  timestamp: string;
}

// ============================================================================
// Risk Management Types
// ============================================================================

/** Risk calculation configuration */
export interface RiskConfig {
  confidenceLevel: number;
  lookbackPeriods: number;
  method: string;
}

/** VaR calculation result */
export interface VaRResult {
  varAmount: number;
  varPercentage: number;
  confidenceLevel: number;
  method: string;
  portfolioValue: number;
}

/** CVaR (Expected Shortfall) result */
export interface CVaRResult {
  cvarAmount: number;
  cvarPercentage: number;
  varAmount: number;
  confidenceLevel: number;
}

/** Drawdown metrics */
export interface DrawdownMetrics {
  maxDrawdown: number;
  maxDrawdownDuration: number;
  currentDrawdown: number;
  recoveryFactor: number;
}

/** Kelly Criterion result for position sizing */
export interface KellyResult {
  kellyFraction: number;
  halfKelly: number;
  quarterKelly: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
}

/** Position sizing recommendation */
export interface PositionSize {
  shares: number;
  dollarAmount: number;
  percentageOfPortfolio: number;
  maxLoss: number;
  reasoning: string;
}

// ============================================================================
// Backtesting Types
// ============================================================================

/** Backtest configuration */
export interface BacktestConfig {
  initialCapital: number;
  startDate: string;
  endDate: string;
  commission: number;
  slippage: number;
  useMarkToMarket: boolean;
}

/** Trade record from backtest */
export interface Trade {
  symbol: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercentage: number;
  commissionPaid: number;
}

/** Backtest performance metrics */
export interface BacktestMetrics {
  totalReturn: number;
  annualReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  finalEquity: number;
}

/** Backtest result */
export interface BacktestResult {
  metrics: BacktestMetrics;
  trades: Trade[];
  equityCurve: number[];
  dates: string[];
}

// ============================================================================
// Market Data Types
// ============================================================================

/** Market data bar/candle */
export interface Bar {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** Real-time quote */
export interface Quote {
  symbol: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  last: number;
  lastSize: number;
  timestamp: string;
}

/** Market data provider configuration */
export interface MarketDataConfig {
  provider: string;
  apiKey?: string;
  apiSecret?: string;
  websocketEnabled: boolean;
}

// ============================================================================
// Strategy Types
// ============================================================================

/** Trading signal from a strategy */
export interface Signal {
  id: string;
  strategyId: string;
  symbol: string;
  direction: string;
  confidence: number;
  entryPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string;
  timestampNs: number;
}

/** Strategy configuration */
export interface StrategyConfig {
  name: string;
  symbols: string[];
  parameters: string;
}

// ============================================================================
// Portfolio Types
// ============================================================================

/** Portfolio position */
export interface Position {
  symbol: string;
  quantity: number;
  avgCost: number;
  marketValue: number;
  unrealizedPnl: number;
  realizedPnl: number;
}

/** Portfolio optimization result */
export interface PortfolioOptimization {
  allocations: Record<string, number>;
  expectedReturn: number;
  risk: number;
  sharpeRatio: number;
}

/** Risk metrics */
export interface RiskMetrics {
  var95: number;
  cvar95: number;
  beta: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

/** Portfolio optimizer configuration */
export interface OptimizerConfig {
  riskFreeRate: number;
  maxPositionSize?: number;
  minPositionSize?: number;
}

// ============================================================================
// JavaScript-Compatible Types (String-based for precision)
// ============================================================================

/** JavaScript-compatible bar data */
export interface JsBar {
  symbol: string;
  timestamp: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

/** JavaScript-compatible signal data */
export interface JsSignal {
  id: string;
  strategyId: string;
  symbol: string;
  direction: string;
  confidence: number;
  entryPrice?: string;
  stopLoss?: string;
  takeProfit?: string;
  quantity?: string;
  reasoning: string;
  timestamp: string;
}

/** JavaScript-compatible order data */
export interface JsOrder {
  id: string;
  symbol: string;
  side: string;
  orderType: string;
  quantity: string;
  limitPrice?: string;
  stopPrice?: string;
  timeInForce: string;
}

/** JavaScript-compatible position data */
export interface JsPosition {
  symbol: string;
  quantity: string;
  avgEntryPrice: string;
  currentPrice: string;
  unrealizedPnl: string;
  side: string;
  marketValue: string;
}

/** JavaScript-compatible configuration */
export interface JsConfig {
  apiKey?: string;
  apiSecret?: string;
  baseUrl?: string;
  paperTrading: boolean;
}

// ============================================================================
// System Types
// ============================================================================

/** Version information */
export interface VersionInfo {
  rustCore: string;
  napiBindings: string;
  rustCompiler: string;
}

/** Generic NAPI result wrapper */
export interface NapiResult {
  success: boolean;
  data?: any;
  error?: string;
}
