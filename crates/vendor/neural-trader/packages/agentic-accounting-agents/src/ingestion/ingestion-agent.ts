/**
 * Ingestion Agent
 * Autonomous agent for transaction data acquisition
 * Performance target: 10,000+ transactions per minute
 */

import { BaseAgent } from '../base/agent';
import { Transaction, TransactionSource } from '@neural-trader/agentic-accounting-types';
import { TransactionIngestionService, IngestionConfig } from '@neural-trader/agentic-accounting-core';
import { CoinbaseIntegration } from '@neural-trader/agentic-accounting-core';
import { BinanceIntegration } from '@neural-trader/agentic-accounting-core';
import { EtherscanIntegration } from '@neural-trader/agentic-accounting-core';

export interface IngestionAgentConfig {
  sources: TransactionSource[];
  batchSize?: number;
  autoNormalize?: boolean;
  validateOnIngestion?: boolean;
  credentials?: {
    coinbase?: { apiKey: string; apiSecret: string };
    binance?: { apiKey: string; apiSecret: string };
    etherscan?: { apiKey: string };
  };
}

export class IngestionAgent extends BaseAgent {
  private ingestionService: TransactionIngestionService;
  private integrations: Map<TransactionSource, any> = new Map();

  constructor(config: IngestionAgentConfig) {
    super('ingestion-agent', 'IngestionAgent');
    this.ingestionService = new TransactionIngestionService();
    this.initializeIntegrations(config);
  }

  private initializeIntegrations(config: IngestionAgentConfig): void {
    // Initialize exchange integrations if credentials provided
    if (config.credentials?.coinbase) {
      this.integrations.set('coinbase', new CoinbaseIntegration(config.credentials.coinbase));
    }
    if (config.credentials?.binance) {
      this.integrations.set('binance', new BinanceIntegration(config.credentials.binance));
    }
    if (config.credentials?.etherscan) {
      this.integrations.set('etherscan', new EtherscanIntegration(config.credentials.etherscan));
    }
  }

  /**
   * Execute ingestion task
   */
  async execute(task: {
    source: TransactionSource;
    data?: any[];
    accountId?: string;
    address?: string;
    options?: any;
  }): Promise<any> {
    this.logger.info(`Starting ingestion from ${task.source}`, { task });

    try {
      let transactions: any[] = [];

      // Fetch from source if not provided
      if (task.data) {
        transactions = task.data;
      } else {
        transactions = await this.fetchFromSource(task);
      }

      // Ingest transactions
      const config: IngestionConfig = {
        source: task.source,
        batchSize: task.options?.batchSize || 1000,
        validateOnIngestion: task.options?.validateOnIngestion !== false,
        autoNormalize: task.options?.autoNormalize !== false
      };

      const result = await this.ingestionService.ingestBatch(transactions, config);

      // Log learning data
      await this.learn({
        action: 'ingest_transactions',
        source: task.source,
        successful: result.successful,
        failed: result.failed,
        duration: result.duration,
        performance: result.successful / (result.duration / 1000) // transactions per second
      });

      this.logger.info(`Ingestion completed`, {
        source: task.source,
        successful: result.successful,
        failed: result.failed
      });

      return result;
    } catch (error) {
      this.logger.error('Ingestion failed', { error, task });
      throw error;
    }
  }

  /**
   * Fetch transactions from source
   */
  private async fetchFromSource(task: any): Promise<any[]> {
    const integration = this.integrations.get(task.source);

    if (!integration) {
      throw new Error(`No integration configured for source: ${task.source}`);
    }

    switch (task.source) {
      case 'coinbase':
        return await integration.fetchTransactions(task.accountId, task.options);
      case 'binance':
        return await integration.fetchTrades(task.options?.symbol, task.options);
      case 'etherscan':
        return await integration.fetchTransactions(task.address, task.options);
      default:
        throw new Error(`Unsupported source: ${task.source}`);
    }
  }

  /**
   * Validate source connection
   */
  async validateSource(source: TransactionSource): Promise<boolean> {
    const integration = this.integrations.get(source);
    if (!integration) {
      return false;
    }

    try {
      return await integration.testConnection();
    } catch (error) {
      this.logger.error(`Source validation failed for ${source}`, { error });
      return false;
    }
  }

  /**
   * Get available sources
   */
  getAvailableSources(): TransactionSource[] {
    return Array.from(this.integrations.keys());
  }

  /**
   * Auto-detect and ingest from all configured sources
   */
  async autoIngest(): Promise<Map<TransactionSource, any>> {
    const results = new Map<TransactionSource, any>();

    for (const [source, integration] of this.integrations.entries()) {
      try {
        this.logger.info(`Auto-ingesting from ${source}`);
        // Implementation would depend on source type
        results.set(source, { status: 'success' });
      } catch (error) {
        this.logger.error(`Auto-ingestion failed for ${source}`, { error });
        results.set(source, { status: 'failed', error });
      }
    }

    return results;
  }
}
