/**
 * Forensic Analysis Agent
 * Autonomous agent for fraud detection and analysis
 * Performance target: <100µs vector queries
 */

import { BaseAgent } from '../base/agent';
import { Transaction } from '@neural-trader/agentic-accounting-types';
import { FraudDetectionService } from '@neural-trader/agentic-accounting-core';
import { MerkleTreeService } from '@neural-trader/agentic-accounting-core';

export interface ForensicAgentConfig {
  sensitivityThreshold?: number;
  autoInvestigate?: boolean;
  generateProofs?: boolean;
}

export class ForensicAgent extends BaseAgent {
  private fraudDetection: FraudDetectionService;
  private merkleTree: MerkleTreeService;
  private config: ForensicAgentConfig;

  constructor(config: ForensicAgentConfig = {}) {
    super('forensic-agent', 'ForensicAgent');
    this.fraudDetection = new FraudDetectionService();
    this.merkleTree = new MerkleTreeService();
    this.config = {
      sensitivityThreshold: config.sensitivityThreshold ?? 0.7,
      autoInvestigate: config.autoInvestigate ?? true,
      generateProofs: config.generateProofs ?? true
    };
  }

  /**
   * Execute forensic analysis task
   */
  async execute(task: {
    action: 'detect_fraud' | 'generate_proof' | 'investigate' | 'analyze_batch';
    transaction?: Transaction;
    transactions?: Transaction[];
  }): Promise<any> {
    this.logger.info(`Executing forensic task: ${task.action}`);

    try {
      switch (task.action) {
        case 'detect_fraud':
          return await this.detectFraud(task.transaction!);
        case 'generate_proof':
          return await this.generateMerkleProof(task.transaction!, task.transactions!);
        case 'investigate':
          return await this.investigate(task.transaction!);
        case 'analyze_batch':
          return await this.analyzeBatch(task.transactions!);
        default:
          throw new Error(`Unknown action: ${task.action}`);
      }
    } catch (error) {
      this.logger.error('Forensic task failed', { error, task });
      throw error;
    }
  }

  /**
   * Detect fraud in transaction
   */
  private async detectFraud(transaction: Transaction): Promise<any> {
    this.logger.info(`Detecting fraud for transaction ${transaction.id}`);

    // Run fraud detection
    const fraudScore = await this.fraudDetection.detectFraud(transaction);

    // Auto-investigate if score exceeds threshold
    if (this.config.autoInvestigate && fraudScore.score >= this.config.sensitivityThreshold) {
      await this.flagForInvestigation(transaction, fraudScore);
    }

    // Log learning data
    await this.learn({
      action: 'detect_fraud',
      transactionId: transaction.id,
      fraudScore: fraudScore.score,
      confidence: fraudScore.confidence,
      patternsMatched: fraudScore.matchedPatterns.length,
      anomalies: fraudScore.anomalies.length
    });

    return fraudScore;
  }

  /**
   * Generate Merkle proof for transaction
   */
  private async generateMerkleProof(
    transaction: Transaction,
    allTransactions: Transaction[]
  ): Promise<any> {
    this.logger.info(`Generating Merkle proof for transaction ${transaction.id}`);

    const proof = this.merkleTree.generateProof(allTransactions, transaction.id);

    // Log for audit trail
    this.logger.debug('Merkle proof generated', {
      transactionId: transaction.id,
      rootHash: proof.rootHash,
      proofLength: proof.proof.length
    });

    return proof;
  }

  /**
   * Investigate suspicious transaction
   */
  private async investigate(transaction: Transaction): Promise<any> {
    this.logger.info(`Investigating transaction ${transaction.id}`);

    // Run comprehensive analysis
    const fraudScore = await this.fraudDetection.detectFraud(transaction);

    // Build investigation report
    const report = {
      transactionId: transaction.id,
      timestamp: new Date(),
      fraudScore: fraudScore.score,
      confidence: fraudScore.confidence,
      findings: {
        matchedPatterns: fraudScore.matchedPatterns,
        anomalies: fraudScore.anomalies,
        riskLevel: this.calculateRiskLevel(fraudScore.score),
        recommendations: this.generateRecommendations(fraudScore)
      },
      investigatedBy: this.name,
      investigationDate: new Date()
    };

    // Log investigation
    await this.learn({
      action: 'investigate',
      transactionId: transaction.id,
      riskLevel: report.findings.riskLevel,
      findings: report.findings.anomalies.length
    });

    return report;
  }

  /**
   * Analyze batch of transactions for fraud
   */
  private async analyzeBatch(transactions: Transaction[]): Promise<any> {
    this.logger.info(`Analyzing batch of ${transactions.length} transactions for fraud`);

    const scores = await this.fraudDetection.detectFraudBatch(transactions);

    // Calculate statistics
    const highRisk = Array.from(scores.values()).filter(
      s => s.score >= this.config.sensitivityThreshold
    );

    const summary = {
      total: transactions.length,
      highRisk: highRisk.length,
      averageScore: Array.from(scores.values()).reduce((sum, s) => sum + s.score, 0) / scores.length,
      timestamp: new Date()
    };

    // Generate Merkle root for batch
    let merkleRoot: string | null = null;
    if (this.config.generateProofs) {
      merkleRoot = this.merkleTree.getRootHash(transactions);
    }

    return {
      scores,
      summary,
      merkleRoot
    };
  }

  /**
   * Flag transaction for investigation
   */
  private async flagForInvestigation(transaction: Transaction, fraudScore: any): Promise<void> {
    this.logger.warn('Transaction flagged for investigation', {
      transactionId: transaction.id,
      fraudScore: fraudScore.score,
      confidence: fraudScore.confidence
    });

    // In production, this would create an investigation case
  }

  /**
   * Calculate risk level from fraud score
   */
  private calculateRiskLevel(score: number): string {
    if (score >= 0.9) return 'CRITICAL';
    if (score >= 0.7) return 'HIGH';
    if (score >= 0.5) return 'MEDIUM';
    if (score >= 0.3) return 'LOW';
    return 'MINIMAL';
  }

  /**
   * Generate recommendations based on fraud score
   */
  private generateRecommendations(fraudScore: any): string[] {
    const recommendations: string[] = [];

    if (fraudScore.score >= 0.9) {
      recommendations.push('Immediate investigation required');
      recommendations.push('Consider blocking future transactions');
      recommendations.push('Notify compliance team');
    } else if (fraudScore.score >= 0.7) {
      recommendations.push('Manual review recommended');
      recommendations.push('Enhanced monitoring for related transactions');
    } else if (fraudScore.score >= 0.5) {
      recommendations.push('Automated monitoring');
      recommendations.push('Periodic review');
    }

    if (fraudScore.anomalies.length > 0) {
      recommendations.push(`Investigate ${fraudScore.anomalies.length} detected anomalies`);
    }

    return recommendations;
  }

  /**
   * Verify Merkle proof
   */
  async verifyProof(transaction: Transaction, proof: any, expectedRootHash: string): Promise<boolean> {
    return this.merkleTree.verifyProof(transaction, proof, expectedRootHash);
  }

  /**
   * Add fraud pattern to database
   */
  async addFraudPattern(pattern: any): Promise<void> {
    await this.fraudDetection.addFraudPattern(pattern);
    this.logger.info(`Added fraud pattern: ${pattern.name}`);
  }
}
