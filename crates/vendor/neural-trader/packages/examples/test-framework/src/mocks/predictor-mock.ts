/**
 * Mock Predictor for testing
 */

import { MockOptions } from '../types';

export interface PredictionRequest {
  features: number[][];
  horizon?: number;
  confidence?: number;
}

export interface PredictionResult {
  predictions: number[];
  confidenceIntervals?: Array<[number, number]>;
  metadata: {
    model: string;
    timestamp: number;
    features: number;
  };
}

/**
 * Mock Predictor implementation
 */
export class MockPredictor {
  private options: MockOptions;
  private callCount: number;
  private model: 'linear' | 'random' | 'constant';

  constructor(options: MockOptions & { model?: 'linear' | 'random' | 'constant' } = {}) {
    this.options = options;
    this.callCount = 0;
    this.model = options.model || 'random';
  }

  /**
   * Make mock predictions
   */
  async predict(request: PredictionRequest): Promise<PredictionResult> {
    this.callCount++;
    await this.simulateDelay();
    this.maybeThrowError();

    const { features, horizon = 1, confidence = 0.95 } = request;
    const predictions = this.generatePredictions(features, horizon);
    const confidenceIntervals = this.generateConfidenceIntervals(
      predictions,
      confidence
    );

    return {
      predictions,
      confidenceIntervals,
      metadata: {
        model: this.model,
        timestamp: Date.now(),
        features: features[0]?.length || 0
      }
    };
  }

  /**
   * Train mock model (no-op)
   */
  async train(data: { X: number[][]; y: number[] }): Promise<void> {
    this.callCount++;
    await this.simulateDelay();
    this.maybeThrowError();
  }

  /**
   * Get call count
   */
  getCallCount(): number {
    return this.callCount;
  }

  /**
   * Reset mock state
   */
  reset(): void {
    this.callCount = 0;
  }

  private generatePredictions(features: number[][], horizon: number): number[] {
    switch (this.model) {
      case 'linear':
        return features.map(f => f.reduce((a, b) => a + b, 0) / f.length);
      case 'constant':
        return Array(features.length).fill(100);
      case 'random':
      default:
        return Array(features.length)
          .fill(0)
          .map(() => Math.random() * 100);
    }
  }

  private generateConfidenceIntervals(
    predictions: number[],
    confidence: number
  ): Array<[number, number]> {
    const z = 1.96; // 95% confidence
    return predictions.map(pred => {
      const margin = pred * 0.1 * z;
      return [pred - margin, pred + margin];
    });
  }

  private async simulateDelay(): Promise<void> {
    if (this.options.delay) {
      await new Promise(resolve => setTimeout(resolve, this.options.delay));
    }
  }

  private maybeThrowError(): void {
    if (this.options.errorRate && Math.random() < this.options.errorRate) {
      throw new Error('Mock Predictor error');
    }
  }
}

/**
 * Create mock predictor
 */
export function createMockPredictor(options: MockOptions = {}): MockPredictor {
  return new MockPredictor(options);
}
