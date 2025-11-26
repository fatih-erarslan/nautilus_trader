import type { AnomalyPoint } from '../index';

/**
 * One-Class SVM for anomaly detection
 *
 * Learns a decision boundary around normal data using an RBF kernel.
 * Points outside the boundary are classified as anomalies.
 *
 * Note: Simplified implementation using stochastic gradient descent.
 * For production, consider using libsvm bindings.
 */
export class OneClassSVM {
  private supportVectors: number[][] = [];
  private alphas: number[] = [];
  private rho = 0; // Decision threshold
  private gamma: number;
  private nu = 0.1; // Fraction of outliers

  constructor(private featureDimensions: number) {
    // RBF kernel parameter (inverse of variance)
    this.gamma = 1 / featureDimensions;
  }

  /**
   * Train One-Class SVM
   */
  async train(data: AnomalyPoint[]): Promise<void> {
    const normalData = data.filter(p => p.label !== 'anomaly');
    const samples = normalData.map(p => p.features);

    // Use a subset as support vectors (simplified)
    const numSupportVectors = Math.min(50, Math.floor(samples.length * 0.2));
    this.supportVectors = this.selectSupportVectors(samples, numSupportVectors);

    // Initialize alphas
    this.alphas = new Array(this.supportVectors.length).fill(1 / this.supportVectors.length);

    // Train using SMO-like algorithm (simplified)
    const maxIterations = 100;
    for (let iter = 0; iter < maxIterations; iter++) {
      for (let i = 0; i < this.supportVectors.length; i++) {
        this.updateAlpha(i, samples);
      }

      if (iter % 10 === 0) {
        const error = this.computeTrainingError(samples);
        console.log(`One-Class SVM Iteration ${iter}, Error: ${error.toFixed(6)}`);
      }
    }

    // Compute threshold rho
    this.rho = this.computeThreshold(samples);
  }

  /**
   * Select support vectors using k-means++
   */
  private selectSupportVectors(samples: number[][], k: number): number[][] {
    const supportVectors: number[][] = [];

    // First support vector: random
    supportVectors.push(samples[Math.floor(Math.random() * samples.length)]);

    // Remaining support vectors: farthest from existing
    for (let i = 1; i < k; i++) {
      let maxDist = -Infinity;
      let farthest: number[] = samples[0];

      for (const sample of samples) {
        const minDistToSupport = Math.min(
          ...supportVectors.map(sv => this.rbfKernel(sample, sv))
        );

        if (minDistToSupport > maxDist) {
          maxDist = minDistToSupport;
          farthest = sample;
        }
      }

      supportVectors.push(farthest);
    }

    return supportVectors;
  }

  /**
   * RBF (Gaussian) kernel
   */
  private rbfKernel(x: number[], y: number[]): number {
    let sumSquares = 0;
    for (let i = 0; i < x.length; i++) {
      const diff = x[i] - y[i];
      sumSquares += diff * diff;
    }
    return Math.exp(-this.gamma * sumSquares);
  }

  /**
   * Update alpha for support vector i
   */
  private updateAlpha(i: number, samples: number[][]): void {
    const learningRate = 0.01;
    const sv = this.supportVectors[i];

    // Compute gradient
    let gradient = 0;
    for (const sample of samples) {
      const kernelValue = this.rbfKernel(sv, sample);
      const prediction = this.decisionFunction(sample);
      gradient += kernelValue * Math.sign(prediction);
    }
    gradient /= samples.length;

    // Update alpha with constraints
    this.alphas[i] -= learningRate * gradient;
    this.alphas[i] = Math.max(0, Math.min(1 / (this.nu * this.supportVectors.length), this.alphas[i]));

    // Normalize alphas
    const sum = this.alphas.reduce((a, b) => a + b, 0);
    this.alphas = this.alphas.map(a => a / sum);
  }

  /**
   * Decision function: f(x) = sum(alpha_i * K(x, sv_i)) - rho
   */
  private decisionFunction(x: number[]): number {
    let score = 0;
    for (let i = 0; i < this.supportVectors.length; i++) {
      score += this.alphas[i] * this.rbfKernel(x, this.supportVectors[i]);
    }
    return score - this.rho;
  }

  /**
   * Compute training error
   */
  private computeTrainingError(samples: number[][]): number {
    let error = 0;
    for (const sample of samples) {
      const decision = this.decisionFunction(sample);
      // Hinge loss
      error += Math.max(0, -decision);
    }
    return error / samples.length;
  }

  /**
   * Compute threshold rho (nu-quantile of decision function)
   */
  private computeThreshold(samples: number[][]): number {
    const scores = samples.map(s => {
      let score = 0;
      for (let i = 0; i < this.supportVectors.length; i++) {
        score += this.alphas[i] * this.rbfKernel(s, this.supportVectors[i]);
      }
      return score;
    });

    scores.sort((a, b) => a - b);
    const quantileIndex = Math.floor(this.nu * scores.length);
    return scores[quantileIndex];
  }

  /**
   * Predict anomaly score (0-1, higher = more anomalous)
   */
  predict(point: AnomalyPoint): number {
    const decision = this.decisionFunction(point.features);

    // Negative decision = anomaly
    // Map to [0, 1] range
    if (decision >= 0) {
      return 0; // Normal
    } else {
      return Math.min(1, -decision / this.rho);
    }
  }
}
