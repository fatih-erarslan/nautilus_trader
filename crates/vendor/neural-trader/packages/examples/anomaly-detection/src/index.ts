/**
 * @neural-trader/example-anomaly-detection
 *
 * Real-time anomaly detection with adaptive thresholds and swarm-based ensemble learning.
 *
 * Features:
 * - Streaming anomaly detection with multiple algorithms
 * - Adaptive threshold learning based on false positive rate
 * - Multi-dimensional anomaly scoring
 * - Conformal prediction for anomaly confidence intervals
 * - Swarm-based ensemble selection and optimization
 * - Memory-persistent pattern library via AgentDB
 *
 * Applications:
 * - Fraud detection in financial transactions
 * - Network intrusion detection
 * - System monitoring and alerting
 * - Trading anomaly detection
 */

export { AnomalyDetector, type AnomalyResult, type DetectorConfig } from './detector';
export { AdaptiveThreshold, type ThresholdConfig } from './adaptive-threshold';
export { EnsembleSwarm, type SwarmConfig, type AlgorithmScore } from './ensemble-swarm';
export { ConformalAnomalyPredictor, type ConformalResult } from './conformal-prediction';

// Algorithm exports
export { IsolationForest } from './algorithms/isolation-forest';
export { LSTMAutoencoder } from './algorithms/lstm-autoencoder';
export { VAE } from './algorithms/vae';
export { OneClassSVM } from './algorithms/one-class-svm';

// Re-export common types
export interface AnomalyPoint {
  timestamp: number;
  features: number[];
  label?: 'normal' | 'anomaly';
  metadata?: Record<string, unknown>;
}

export interface DetectionResult {
  isAnomaly: boolean;
  score: number;
  confidence: number;
  algorithm: string;
  explanation?: string;
  conformalInterval?: [number, number];
}
