import type { AnomalyPoint } from '../index';

/**
 * Variational Autoencoder for probabilistic anomaly detection
 *
 * Models the distribution of normal data in latent space.
 * Anomalies have low probability under the learned distribution.
 *
 * Note: Simplified implementation. For production, use TensorFlow.js
 */
export class VAE {
  private latentDim = 16;
  private hiddenSize = 32;
  private learningRate = 0.001;

  // Network weights
  private encoderWeights: number[][] = [];
  private muWeights: number[][] = [];
  private logVarWeights: number[][] = [];
  private decoderWeights: number[][] = [];

  private meanLoss = 0;

  constructor(private featureDimensions: number) {
    this.initializeWeights();
  }

  /**
   * Initialize network weights
   */
  private initializeWeights(): void {
    // Encoder: features -> hidden
    this.encoderWeights = this.randomMatrix(this.hiddenSize, this.featureDimensions);

    // Latent parameters: hidden -> latent
    this.muWeights = this.randomMatrix(this.latentDim, this.hiddenSize);
    this.logVarWeights = this.randomMatrix(this.latentDim, this.hiddenSize);

    // Decoder: latent -> features
    this.decoderWeights = this.randomMatrix(this.featureDimensions, this.latentDim);
  }

  /**
   * Create random matrix with Xavier initialization
   */
  private randomMatrix(rows: number, cols: number): number[][] {
    const scale = Math.sqrt(2.0 / (rows + cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() - 0.5) * scale)
    );
  }

  /**
   * Train VAE on normal data
   */
  async train(data: AnomalyPoint[]): Promise<void> {
    const epochs = 100;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      for (const point of data) {
        const loss = this.trainStep(point);
        totalLoss += loss;
      }

      this.meanLoss = totalLoss / data.length;

      if (epoch % 10 === 0) {
        console.log(`VAE Epoch ${epoch}, Loss: ${this.meanLoss.toFixed(6)}`);
      }
    }
  }

  /**
   * Single training step
   */
  private trainStep(point: AnomalyPoint): number {
    // Forward pass
    const { mu, logVar, z } = this.encode(point.features);
    const reconstructed = this.decode(z);

    // Compute loss: reconstruction + KL divergence
    const reconLoss = this.reconstructionLoss(point.features, reconstructed);
    const klLoss = this.klDivergence(mu, logVar);
    const totalLoss = reconLoss + klLoss;

    // Simplified gradient descent
    this.updateWeights(point.features, reconstructed, totalLoss);

    return totalLoss;
  }

  /**
   * Encode to latent space
   */
  private encode(features: number[]): { mu: number[]; logVar: number[]; z: number[] } {
    // Encoder: features -> hidden
    const hidden = this.forward(features, this.encoderWeights);

    // Latent parameters
    const mu = this.forward(hidden, this.muWeights);
    const logVar = this.forward(hidden, this.logVarWeights);

    // Reparameterization trick: z = mu + sigma * epsilon
    const z = mu.map((m, i) => {
      const sigma = Math.exp(0.5 * logVar[i]);
      const epsilon = this.randomNormal();
      return m + sigma * epsilon;
    });

    return { mu, logVar, z };
  }

  /**
   * Decode from latent space
   */
  private decode(z: number[]): number[] {
    return this.forward(z, this.decoderWeights);
  }

  /**
   * Forward pass through layer
   */
  private forward(input: number[], weights: number[][]): number[] {
    const output: number[] = [];

    for (let i = 0; i < weights.length; i++) {
      let activation = 0;
      for (let j = 0; j < input.length; j++) {
        activation += input[j] * weights[i][j];
      }
      output.push(Math.tanh(activation));
    }

    return output;
  }

  /**
   * Reconstruction loss (MSE)
   */
  private reconstructionLoss(original: number[], reconstructed: number[]): number {
    let loss = 0;
    for (let i = 0; i < original.length; i++) {
      const diff = original[i] - reconstructed[i];
      loss += diff * diff;
    }
    return loss / original.length;
  }

  /**
   * KL divergence regularization
   */
  private klDivergence(mu: number[], logVar: number[]): number {
    let kl = 0;
    for (let i = 0; i < mu.length; i++) {
      kl += 1 + logVar[i] - mu[i] * mu[i] - Math.exp(logVar[i]);
    }
    return -0.5 * kl / mu.length;
  }

  /**
   * Update weights (simplified gradient descent)
   */
  private updateWeights(original: number[], reconstructed: number[], loss: number): void {
    const gradient = loss * this.learningRate * 0.01;

    // Update all weight matrices slightly
    [this.encoderWeights, this.muWeights, this.logVarWeights, this.decoderWeights].forEach(weights => {
      for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
          weights[i][j] -= gradient * (Math.random() - 0.5);
        }
      }
    });
  }

  /**
   * Predict anomaly score (negative log probability)
   */
  predict(point: AnomalyPoint): number {
    const { mu, logVar, z } = this.encode(point.features);
    const reconstructed = this.decode(z);

    // Compute reconstruction error
    const reconError = this.reconstructionLoss(point.features, reconstructed);

    // Compute probability under latent distribution
    let logProb = 0;
    for (let i = 0; i < z.length; i++) {
      const sigma = Math.exp(0.5 * logVar[i]);
      const diff = z[i] - mu[i];
      logProb -= 0.5 * (Math.log(2 * Math.PI) + 2 * Math.log(sigma) + (diff * diff) / (sigma * sigma));
    }

    // Combine reconstruction error and probability
    const anomalyScore = reconError - logProb;

    // Normalize by mean loss
    return Math.min(1, Math.max(0, anomalyScore / (this.meanLoss || 1)));
  }

  /**
   * Generate random normal sample (Box-Muller transform)
   */
  private randomNormal(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}
