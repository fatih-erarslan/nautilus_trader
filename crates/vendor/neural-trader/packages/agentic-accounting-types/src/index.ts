/**
 * Agentic Accounting Types
 *
 * Shared type definitions for the accounting system
 */

/**
 * Transaction represents a financial transaction (buy, sell, trade, etc.)
 */
export interface Transaction {
  id: string;
  timestamp: Date;
  type: 'BUY' | 'SELL' | 'TRADE' | 'CONVERT' | 'INCOME' | 'DIVIDEND' | 'FEE' | 'TRANSFER';
  asset: string;
  quantity: number;
  price: number;
  fees?: number;
  exchange?: string;
  walletAddress?: string;
  metadata?: Record<string, any>;
  source?: string | TransactionSource;
}

/**
 * Position represents holdings of a specific asset
 */
export interface Position {
  id: string;
  asset: string;
  quantity: number;
  averageCost: number;
  currentValue: number;
  unrealizedGainLoss: number;
  lots: Lot[];
  lastUpdated: Date;
  totalCost: number;
  averageCostBasis: number;
}

/**
 * Lot represents a specific purchase lot for tax accounting
 */
export interface Lot {
  id: string;
  asset: string;
  quantity: number;
  purchasePrice: number;
  purchaseDate: Date;
  acquisitionDate: Date; // Alias for purchaseDate
  transactionId: string;
  disposed?: boolean;
  disposedDate?: Date;
  disposedPrice?: number;
  isOpen: boolean;
  remainingQuantity: number;
  costBasis: number;
}

/**
 * Tax calculation result
 */
export interface TaxResult {
  totalGain: number;
  totalLoss: number;
  shortTermGain: number;
  shortTermLoss: number;
  longTermGain: number;
  longTermLoss: number;
  transactions: TaxTransaction[];
  year: number;
}

/**
 * Tax transaction with gain/loss calculation
 */
export interface TaxTransaction {
  id: string;
  asset: string;
  buyDate: Date;
  sellDate: Date;
  acquisitionDate: Date; // Alias for buyDate
  disposalDate: Date; // Alias for sellDate
  quantity: number;
  costBasis: number;
  proceeds: number;
  gainLoss: number;
  holdingPeriod: number;
  type: 'short-term' | 'long-term';
  isLongTerm: boolean;
  washSaleAdjustment?: number;
  method?: string;
  metadata?: Record<string, any>;
}

/**
 * Transaction source (exchange, wallet, etc.)
 */
export interface TransactionSource {
  type: 'exchange' | 'wallet' | 'csv' | 'api';
  name: string;
  credentials?: Record<string, any>;
}

/**
 * Result of transaction ingestion
 */
export interface IngestionResult {
  success: boolean;
  transactionsImported: number;
  errors: string[];
  warnings: string[];
  source: TransactionSource;
  timestamp: Date;
  total?: number;
  duration?: number;
  successful?: number;
  failed?: number;
  transactions?: Transaction[];
}

/**
 * Compliance rule
 */
export interface ComplianceRule {
  id: string;
  name: string;
  description: string;
  category: 'tax' | 'regulatory' | 'reporting';
  jurisdiction: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
}

/**
 * Compliance violation
 */
export interface ComplianceViolation {
  ruleId: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  transactionId?: string;
  details?: Record<string, any>;
  timestamp: Date;
}

/**
 * Agent configuration
 */
export interface AgentConfig {
  agentId: string;
  agentType: string;
  enableLearning?: boolean;
  enableMetrics?: boolean;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

/**
 * Placeholder for future type definitions
 */
export interface AccountingTypes {
  // Additional types will be defined here
}
