/**
 * Fraud Detection Example
 *
 * Demonstrates real-time credit card fraud detection using anomaly detection
 * with adaptive thresholds and ensemble learning.
 */

import { AnomalyDetector } from '../src/detector';
import type { AnomalyPoint } from '../src/index';

interface Transaction {
  id: string;
  amount: number;
  location: string;
  timestamp: number;
  merchantCategory: string;
  cardPresent: boolean;
}

class FraudDetectionSystem {
  private detector: AnomalyDetector;
  private userProfiles: Map<string, TransactionProfile> = new Map();

  constructor() {
    this.detector = new AnomalyDetector({
      targetFalsePositiveRate: 0.01, // Very low FPR for fraud
      featureDimensions: 5,
      useEnsemble: true, // Use multiple algorithms
      useConformal: true, // Get confidence intervals
      windowSize: 1000,
      minCalibrationSamples: 100,
      agentDbPath: './fraud-patterns.db',
    });
  }

  /**
   * Initialize detector with historical transaction data
   */
  async initialize(historicalTransactions: Transaction[]): Promise<void> {
    console.log('Initializing fraud detection system...');

    // Build user profiles
    for (const tx of historicalTransactions) {
      this.updateUserProfile(tx);
    }

    // Convert to anomaly points
    const trainingData: AnomalyPoint[] = historicalTransactions.map(tx =>
      this.transactionToAnomalyPoint(tx, 'normal')
    );

    // Calibrate detector
    await this.detector.calibrate(trainingData);

    console.log('Fraud detection system ready');
  }

  /**
   * Check if a transaction is fraudulent
   */
  async checkTransaction(tx: Transaction): Promise<{
    isFraud: boolean;
    riskScore: number;
    confidence: number;
    reason: string;
  }> {
    const point = this.transactionToAnomalyPoint(tx);
    const result = await this.detector.detect(point);

    const reason = this.explainDetection(tx, result.detection.score);

    return {
      isFraud: result.detection.isAnomaly,
      riskScore: result.detection.score,
      confidence: result.detection.confidence,
      reason,
    };
  }

  /**
   * Convert transaction to feature vector
   */
  private transactionToAnomalyPoint(
    tx: Transaction,
    label?: 'normal' | 'anomaly'
  ): AnomalyPoint {
    const profile = this.userProfiles.get(tx.id) || this.createDefaultProfile();

    // Feature engineering
    const features = [
      // 1. Amount deviation from user average
      (tx.amount - profile.averageAmount) / (profile.stdAmount || 1),

      // 2. Time since last transaction (normalized)
      Math.min(10, (tx.timestamp - profile.lastTransactionTime) / (1000 * 60 * 60)), // hours

      // 3. Location distance from usual (simplified: 0=same, 1=different)
      tx.location === profile.mostCommonLocation ? 0 : 1,

      // 4. Merchant category familiarity (0=familiar, 1=new)
      profile.merchantCategories.has(tx.merchantCategory) ? 0 : 1,

      // 5. Card not present risk (online transactions riskier)
      tx.cardPresent ? 0 : 1,
    ];

    return {
      timestamp: tx.timestamp,
      features,
      label,
      metadata: {
        transactionId: tx.id,
        amount: tx.amount,
        location: tx.location,
        merchantCategory: tx.merchantCategory,
      },
    };
  }

  /**
   * Update user transaction profile
   */
  private updateUserProfile(tx: Transaction): void {
    let profile = this.userProfiles.get(tx.id);

    if (!profile) {
      profile = this.createDefaultProfile();
      this.userProfiles.set(tx.id, profile);
    }

    // Update statistics
    profile.transactionCount++;
    profile.totalAmount += tx.amount;
    profile.averageAmount = profile.totalAmount / profile.transactionCount;

    // Update standard deviation (simplified)
    profile.stdAmount = Math.sqrt(
      ((tx.amount - profile.averageAmount) ** 2 + profile.stdAmount ** 2) / 2
    );

    profile.lastTransactionTime = tx.timestamp;
    profile.merchantCategories.add(tx.merchantCategory);

    // Update most common location
    profile.locationCounts.set(
      tx.location,
      (profile.locationCounts.get(tx.location) || 0) + 1
    );

    const maxLocation = Array.from(profile.locationCounts.entries()).reduce((a, b) =>
      b[1] > a[1] ? b : a
    );
    profile.mostCommonLocation = maxLocation[0];
  }

  /**
   * Explain why transaction was flagged
   */
  private explainDetection(tx: Transaction, score: number): string {
    const reasons: string[] = [];

    const profile = this.userProfiles.get(tx.id);
    if (!profile) return 'Unknown user';

    if (Math.abs(tx.amount - profile.averageAmount) > 3 * profile.stdAmount) {
      reasons.push('Unusual transaction amount');
    }

    if (tx.location !== profile.mostCommonLocation) {
      reasons.push('Transaction from unusual location');
    }

    if (!profile.merchantCategories.has(tx.merchantCategory)) {
      reasons.push('New merchant category');
    }

    if (!tx.cardPresent) {
      reasons.push('Card-not-present transaction');
    }

    const timeSinceLastTx = (tx.timestamp - profile.lastTransactionTime) / (1000 * 60);
    if (timeSinceLastTx < 5) {
      reasons.push('Rapid succession of transactions');
    }

    return reasons.length > 0 ? reasons.join(', ') : 'Pattern deviation detected';
  }

  /**
   * Provide feedback to improve detection
   */
  async reportFraud(transactionId: string, isActualFraud: boolean): Promise<void> {
    // Find transaction in history
    const stats = this.detector.getStatistics();
    const recentDetections = stats.totalDetections;

    // Provide feedback (would need to track timestamps better in production)
    console.log(`Feedback received: Transaction ${transactionId} was ${isActualFraud ? 'fraudulent' : 'legitimate'}`);
  }

  private createDefaultProfile(): TransactionProfile {
    return {
      transactionCount: 0,
      totalAmount: 0,
      averageAmount: 0,
      stdAmount: 0,
      lastTransactionTime: 0,
      mostCommonLocation: '',
      merchantCategories: new Set(),
      locationCounts: new Map(),
    };
  }
}

interface TransactionProfile {
  transactionCount: number;
  totalAmount: number;
  averageAmount: number;
  stdAmount: number;
  lastTransactionTime: number;
  mostCommonLocation: string;
  merchantCategories: Set<string>;
  locationCounts: Map<string, number>;
}

// Example usage
async function main() {
  const fraudDetector = new FraudDetectionSystem();

  // Generate synthetic historical data
  const historicalTransactions: Transaction[] = Array.from({ length: 200 }, (_, i) => ({
    id: 'user_123',
    amount: 50 + Math.random() * 100,
    location: Math.random() > 0.1 ? 'New York' : 'Boston',
    timestamp: Date.now() - (200 - i) * 60 * 60 * 1000,
    merchantCategory: ['grocery', 'gas', 'restaurant'][Math.floor(Math.random() * 3)],
    cardPresent: Math.random() > 0.2,
  }));

  await fraudDetector.initialize(historicalTransactions);

  // Check suspicious transactions
  const suspiciousTransactions: Transaction[] = [
    {
      id: 'user_123',
      amount: 5000, // Very large amount
      location: 'Tokyo', // Unusual location
      timestamp: Date.now(),
      merchantCategory: 'electronics',
      cardPresent: false,
    },
    {
      id: 'user_123',
      amount: 75, // Normal amount
      location: 'New York',
      timestamp: Date.now() + 1000,
      merchantCategory: 'grocery',
      cardPresent: true,
    },
  ];

  for (const tx of suspiciousTransactions) {
    const result = await fraudDetector.checkTransaction(tx);

    console.log('\nTransaction:', tx.id);
    console.log('Amount:', tx.amount);
    console.log('Location:', tx.location);
    console.log('Fraud Risk:', result.isFraud ? 'HIGH' : 'LOW');
    console.log('Risk Score:', result.riskScore.toFixed(3));
    console.log('Confidence:', (result.confidence * 100).toFixed(1) + '%');
    console.log('Reason:', result.reason);
  }
}

// Run example
if (require.main === module) {
  main().catch(console.error);
}

export { FraudDetectionSystem };
