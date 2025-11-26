/**
 * Fraud Detection System
 * Vector-based fraud pattern detection using AgentDB
 * Performance target: <100µs queries
 */

import { Transaction } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';

export interface FraudPattern {
  id: string;
  name: string;
  description: string;
  vector: number[];
  severity: number;
  metadata?: any;
}

export interface FraudScore {
  transactionId: string;
  score: number;
  confidence: number;
  matchedPatterns: FraudPattern[];
  anomalies: string[];
  timestamp: Date;
}

interface VectorDB {
  createCollection: (name: string, options: any) => Promise<void>;
  query: (collection: string, options: any) => Promise<any[]>;
  insert: (collection: string, data: any) => Promise<void>;
}

export class FraudDetectionService {
  private vectorDB: VectorDB;
  private initialized: boolean = false;
  private patterns: Map<string, FraudPattern> = new Map();

  constructor() {
    // TODO: Initialize actual AgentDB when available
    // For now, use in-memory placeholder
    this.vectorDB = this.createPlaceholderDB();
  }

  /**
   * Initialize fraud pattern database
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Create collection for fraud patterns
      await this.vectorDB.createCollection('fraud_patterns', {
        quantization: 'binary',
        hnsw: true
      });

      // Load known fraud patterns
      await this.loadFraudPatterns();

      this.initialized = true;
      logger.info('Fraud detection system initialized');
    } catch (error) {
      logger.error('Failed to initialize fraud detection', { error });
      throw error;
    }
  }

  /**
   * Detect fraudulent patterns in transaction
   * Performance target: <100µs per query
   */
  async detectFraud(transaction: Transaction): Promise<FraudScore> {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = process.hrtime.bigint();

    try {
      // Generate transaction vector
      const vector = this.generateTransactionVector(transaction);

      // Query similar fraud patterns
      const similarPatterns = await this.vectorDB.query('fraud_patterns', {
        vector,
        topK: 5,
        threshold: 0.7
      });

      // Calculate fraud score
      const score = this.calculateFraudScore(transaction, similarPatterns);

      const endTime = process.hrtime.bigint();
      const duration = Number(endTime - startTime) / 1000; // Convert to microseconds

      logger.debug(`Fraud detection completed in ${duration.toFixed(2)}µs`, {
        score: score.score,
        matches: similarPatterns.length
      });

      return score;
    } catch (error) {
      logger.error('Fraud detection failed', { error, transaction });
      throw error;
    }
  }

  /**
   * Generate vector representation of transaction
   */
  private generateTransactionVector(transaction: Transaction): number[] {
    // Convert transaction features to vector
    // Features: amount, time of day, day of week, asset, type, etc.
    const vector: number[] = [];

    // Amount (normalized log scale)
    const amount = transaction.quantity * transaction.price;
    vector.push(Math.log10(amount + 1) / 10);

    // Time features
    const hour = transaction.timestamp.getHours() / 24;
    const dayOfWeek = transaction.timestamp.getDay() / 7;
    vector.push(hour, dayOfWeek);

    // Transaction type (one-hot encoded)
    const types = ['BUY', 'SELL', 'TRADE', 'CONVERT', 'INCOME', 'DIVIDEND', 'FEE', 'TRANSFER'];
    types.forEach(type => {
      vector.push(transaction.type === type ? 1 : 0);
    });

    // Fees ratio
    const feeRatio = transaction.fees ? transaction.fees / amount : 0;
    vector.push(feeRatio);

    // Pad to 128 dimensions for HNSW optimization
    while (vector.length < 128) {
      vector.push(0);
    }

    return vector;
  }

  /**
   * Calculate fraud score from similar patterns
   */
  private calculateFraudScore(
    transaction: Transaction,
    similarPatterns: any[]
  ): FraudScore {
    let totalScore = 0;
    let totalConfidence = 0;
    const matchedPatterns: FraudPattern[] = [];
    const anomalies: string[] = [];

    for (const match of similarPatterns) {
      const pattern = match.metadata as FraudPattern;
      const similarity = match.score || 0;

      if (similarity > 0.7) {
        totalScore += pattern.severity * similarity;
        totalConfidence += similarity;
        matchedPatterns.push(pattern);
        anomalies.push(`Similar to ${pattern.name} (${(similarity * 100).toFixed(1)}% match)`);
      }
    }

    // Normalize score to 0-1 range
    const normalizedScore = matchedPatterns.length > 0
      ? totalScore / matchedPatterns.length
      : 0;

    const confidence = matchedPatterns.length > 0
      ? totalConfidence / matchedPatterns.length
      : 0;

    // Check for statistical anomalies
    this.detectAnomalies(transaction, anomalies);

    return {
      transactionId: transaction.id,
      score: normalizedScore,
      confidence,
      matchedPatterns,
      anomalies,
      timestamp: new Date()
    };
  }

  /**
   * Detect statistical anomalies
   */
  private detectAnomalies(transaction: Transaction, anomalies: string[]): void {
    const amount = transaction.quantity * transaction.price;

    // Unusually high amount
    if (amount > 1000000) {
      anomalies.push('Unusually high transaction amount');
    }

    // Off-hours trading
    const hour = transaction.timestamp.getHours();
    if (hour < 6 || hour > 22) {
      anomalies.push('Transaction outside normal hours');
    }

    // Weekend trading (crypto is 24/7, but some patterns are suspicious)
    const day = transaction.timestamp.getDay();
    if (day === 0 || day === 6) {
      anomalies.push('Weekend transaction');
    }

    // High fee ratio
    if (transaction.fees && transaction.fees / amount > 0.1) {
      anomalies.push('Unusually high fee ratio');
    }
  }

  /**
   * Add new fraud pattern to database
   */
  async addFraudPattern(pattern: FraudPattern): Promise<void> {
    await this.vectorDB.insert('fraud_patterns', {
      vector: pattern.vector,
      metadata: pattern
    });

    logger.info(`Added fraud pattern: ${pattern.name}`);
  }

  /**
   * Load known fraud patterns
   */
  private async loadFraudPatterns(): Promise<void> {
    // Load pre-defined fraud patterns
    const patterns: FraudPattern[] = [
      {
        id: 'structuring',
        name: 'Transaction Structuring',
        description: 'Multiple transactions just below reporting threshold',
        vector: this.generatePatternVector({ type: 'structuring', amount: 9999 }),
        severity: 0.9
      },
      {
        id: 'round-tripping',
        name: 'Round Tripping',
        description: 'Buying and selling same asset within short timeframe',
        vector: this.generatePatternVector({ type: 'round-trip', duration: 60 }),
        severity: 0.7
      },
      {
        id: 'layering',
        name: 'Layering',
        description: 'Multiple rapid trades to obscure origin',
        vector: this.generatePatternVector({ type: 'layering', frequency: 'high' }),
        severity: 0.8
      }
    ];

    for (const pattern of patterns) {
      await this.addFraudPattern(pattern);
    }
  }

  /**
   * Generate vector for fraud pattern
   */
  private generatePatternVector(config: any): number[] {
    const vector = new Array(128).fill(0);

    // Simplified pattern vector generation
    if (config.type === 'structuring') {
      vector[0] = 0.95; // High amount feature
      vector[1] = 0.5;  // Time feature
    } else if (config.type === 'round-trip') {
      vector[2] = 0.8;  // Quick turnaround
      vector[3] = 0.9;  // Same asset
    } else if (config.type === 'layering') {
      vector[4] = 0.9;  // High frequency
      vector[5] = 0.8;  // Multiple transactions
    }

    return vector;
  }

  /**
   * Batch fraud detection
   */
  async detectFraudBatch(transactions: Transaction[]): Promise<Map<string, FraudScore>> {
    const results = new Map<string, FraudScore>();

    await Promise.all(
      transactions.map(async tx => {
        const score = await this.detectFraud(tx);
        results.set(tx.id, score);
      })
    );

    return results;
  }

  /**
   * Create placeholder VectorDB for development
   * TODO: Replace with actual AgentDB implementation
   */
  private createPlaceholderDB(): VectorDB {
    return {
      createCollection: async (name: string, options: any) => {
        logger.debug(`Placeholder: Created collection ${name}`);
      },
      query: async (collection: string, options: any) => {
        // Return empty results for now
        return [];
      },
      insert: async (collection: string, data: any) => {
        // Store pattern in memory
        if (data.metadata && data.metadata.id) {
          this.patterns.set(data.metadata.id, data.metadata);
        }
        logger.debug(`Placeholder: Inserted into ${collection}`);
      }
    };
  }
}
