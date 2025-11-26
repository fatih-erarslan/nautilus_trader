import { AgentDB } from 'agentdb';
import { AdaptiveThreshold } from './adaptive-threshold';
import { EnsembleSwarm } from './ensemble-swarm';
import { ConformalAnomalyPredictor } from './conformal-prediction';
import type { AnomalyPoint, DetectionResult } from './index';

export interface DetectorConfig {
  /**
   * Desired false positive rate for adaptive thresholds (0-1)
   * Lower values = stricter anomaly detection
   */
  targetFalsePositiveRate: number;

  /**
   * Number of dimensions in the feature space
   */
  featureDimensions: number;

  /**
   * Enable swarm-based ensemble selection
   */
  useEnsemble: boolean;

  /**
   * Enable conformal prediction for confidence intervals
   */
  useConformal: boolean;

  /**
   * AgentDB configuration for pattern persistence
   */
  agentDbPath?: string;

  /**
   * OpenRouter API key for contextual interpretation
   */
  openRouterApiKey?: string;

  /**
   * Sliding window size for streaming detection
   */
  windowSize: number;

  /**
   * Minimum calibration samples before detection
   */
  minCalibrationSamples: number;
}

export interface AnomalyResult {
  point: AnomalyPoint;
  detection: DetectionResult;
  timestamp: number;
}

/**
 * Real-time anomaly detector with adaptive thresholds and ensemble learning
 */
export class AnomalyDetector {
  private adaptiveThreshold: AdaptiveThreshold;
  private ensembleSwarm?: EnsembleSwarm;
  private conformalPredictor?: ConformalAnomalyPredictor;
  private agentDb?: AgentDB;
  private calibrationBuffer: AnomalyPoint[] = [];
  private detectionHistory: AnomalyResult[] = [];
  private isCalibrated = false;

  constructor(private config: DetectorConfig) {
    this.adaptiveThreshold = new AdaptiveThreshold({
      targetFalsePositiveRate: config.targetFalsePositiveRate,
      windowSize: config.windowSize,
      adaptationRate: 0.1,
    });

    if (config.useEnsemble) {
      this.ensembleSwarm = new EnsembleSwarm({
        featureDimensions: config.featureDimensions,
        populationSize: 20,
        maxGenerations: 100,
        crossoverRate: 0.8,
        mutationRate: 0.1,
      });
    }

    if (config.useConformal) {
      this.conformalPredictor = new ConformalAnomalyPredictor({
        alpha: 0.1, // 90% confidence
        windowSize: 1000,
      });
    }

    if (config.agentDbPath) {
      this.initializeAgentDB(config.agentDbPath);
    }
  }

  /**
   * Initialize AgentDB for persistent pattern storage
   */
  private async initializeAgentDB(dbPath: string): Promise<void> {
    this.agentDb = new AgentDB({
      path: dbPath,
      dimensions: this.config.featureDimensions,
    });

    // Load historical patterns
    await this.loadPatterns();
  }

  /**
   * Load historical anomaly patterns from AgentDB
   */
  private async loadPatterns(): Promise<void> {
    if (!this.agentDb) return;

    try {
      // Query recent anomaly patterns
      const patterns = await this.agentDb.query({
        vector: new Array(this.config.featureDimensions).fill(0),
        k: 100,
        filter: { type: 'anomaly' },
      });

      console.log(`Loaded ${patterns.length} historical anomaly patterns`);
    } catch (error) {
      console.error('Failed to load patterns:', error);
    }
  }

  /**
   * Store anomaly pattern in AgentDB for future reference
   */
  private async storePattern(result: AnomalyResult): Promise<void> {
    if (!this.agentDb || !result.detection.isAnomaly) return;

    try {
      await this.agentDb.insert({
        id: `anomaly_${result.timestamp}`,
        vector: result.point.features,
        metadata: {
          type: 'anomaly',
          score: result.detection.score,
          confidence: result.detection.confidence,
          algorithm: result.detection.algorithm,
          timestamp: result.timestamp,
          ...result.point.metadata,
        },
      });
    } catch (error) {
      console.error('Failed to store pattern:', error);
    }
  }

  /**
   * Calibrate the detector with labeled training data
   */
  async calibrate(trainingData: AnomalyPoint[]): Promise<void> {
    if (trainingData.length < this.config.minCalibrationSamples) {
      throw new Error(
        `Insufficient calibration data: need ${this.config.minCalibrationSamples}, got ${trainingData.length}`
      );
    }

    console.log(`Calibrating with ${trainingData.length} samples...`);

    // Train ensemble if enabled
    if (this.ensembleSwarm) {
      await this.ensembleSwarm.train(trainingData);
    }

    // Calibrate conformal predictor
    if (this.conformalPredictor) {
      await this.conformalPredictor.calibrate(trainingData);
    }

    // Initialize adaptive thresholds
    for (const point of trainingData) {
      const baseScore = this.computeBaseScore(point);
      this.adaptiveThreshold.update(baseScore, point.label === 'anomaly');
    }

    this.isCalibrated = true;
    console.log('Calibration complete');
  }

  /**
   * Detect anomalies in streaming data
   */
  async detect(point: AnomalyPoint): Promise<AnomalyResult> {
    if (!this.isCalibrated) {
      // Buffer points for calibration
      this.calibrationBuffer.push(point);

      if (this.calibrationBuffer.length >= this.config.minCalibrationSamples) {
        await this.calibrate(this.calibrationBuffer);
        this.calibrationBuffer = [];
      }

      return {
        point,
        detection: {
          isAnomaly: false,
          score: 0,
          confidence: 0,
          algorithm: 'calibrating',
        },
        timestamp: Date.now(),
      };
    }

    // Compute anomaly scores
    const scores = await this.computeScores(point);

    // Get ensemble score if available
    const finalScore = this.ensembleSwarm
      ? this.ensembleSwarm.predictEnsemble(point)
      : scores.isolationForest; // Fallback to single algorithm

    // Check adaptive threshold
    const threshold = this.adaptiveThreshold.getThreshold();
    const isAnomaly = finalScore > threshold;

    // Get conformal confidence interval if available
    let conformalInterval: [number, number] | undefined;
    let confidence = 0;

    if (this.conformalPredictor) {
      const conformalResult = this.conformalPredictor.predict(point);
      conformalInterval = conformalResult.interval;
      confidence = conformalResult.confidence;
    } else {
      // Simple confidence based on score distance from threshold
      confidence = Math.min(1, Math.abs(finalScore - threshold) / threshold);
    }

    // Update adaptive threshold with feedback
    this.adaptiveThreshold.update(finalScore, isAnomaly);

    const result: AnomalyResult = {
      point,
      detection: {
        isAnomaly,
        score: finalScore,
        confidence,
        algorithm: this.ensembleSwarm ? 'ensemble' : 'isolation-forest',
        conformalInterval,
      },
      timestamp: Date.now(),
    };

    // Store in history
    this.detectionHistory.push(result);
    if (this.detectionHistory.length > this.config.windowSize) {
      this.detectionHistory.shift();
    }

    // Persist anomaly patterns
    await this.storePattern(result);

    return result;
  }

  /**
   * Compute base anomaly score (simple distance-based)
   */
  private computeBaseScore(point: AnomalyPoint): number {
    // Simple Euclidean distance from origin as baseline
    const sumSquares = point.features.reduce((sum, f) => sum + f * f, 0);
    return Math.sqrt(sumSquares);
  }

  /**
   * Compute scores from all algorithms
   */
  private async computeScores(point: AnomalyPoint): Promise<Record<string, number>> {
    const scores: Record<string, number> = {
      isolationForest: this.computeBaseScore(point),
    };

    // Add ensemble scores if available
    if (this.ensembleSwarm) {
      const ensembleScores = this.ensembleSwarm.getAlgorithmScores(point);
      Object.assign(scores, ensembleScores);
    }

    return scores;
  }

  /**
   * Provide feedback to improve detection accuracy
   */
  async provideFeedback(timestamp: number, isActualAnomaly: boolean): Promise<void> {
    const result = this.detectionHistory.find(r => r.timestamp === timestamp);
    if (!result) {
      console.warn(`No detection found for timestamp ${timestamp}`);
      return;
    }

    // Update adaptive threshold with ground truth
    const wasPredictedAnomaly = result.detection.isAnomaly;
    const isFalsePositive = wasPredictedAnomaly && !isActualAnomaly;
    const isFalseNegative = !wasPredictedAnomaly && isActualAnomaly;

    if (isFalsePositive || isFalseNegative) {
      // Adjust threshold more aggressively for errors
      this.adaptiveThreshold.update(result.detection.score, isActualAnomaly, 2.0);
    }

    // Retrain ensemble periodically with feedback
    if (this.ensembleSwarm && this.detectionHistory.length % 100 === 0) {
      const feedbackData = this.detectionHistory.map(r => ({
        ...r.point,
        label: r.detection.isAnomaly ? 'anomaly' as const : 'normal' as const,
      }));

      await this.ensembleSwarm.train(feedbackData);
    }
  }

  /**
   * Get current detection statistics
   */
  getStatistics() {
    const recentDetections = this.detectionHistory.slice(-this.config.windowSize);
    const anomalyCount = recentDetections.filter(r => r.detection.isAnomaly).length;

    return {
      totalDetections: this.detectionHistory.length,
      recentAnomalies: anomalyCount,
      anomalyRate: anomalyCount / Math.max(1, recentDetections.length),
      currentThreshold: this.adaptiveThreshold.getThreshold(),
      isCalibrated: this.isCalibrated,
    };
  }
}
