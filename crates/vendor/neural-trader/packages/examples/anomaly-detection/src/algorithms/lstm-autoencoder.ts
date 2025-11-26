import type { AnomalyPoint } from '../index';

/**
 * LSTM Autoencoder for sequence anomaly detection
 *
 * Learns to reconstruct normal sequences. High reconstruction error
 * indicates anomalous patterns.
 *
 * Note: This is a simplified implementation. For production, consider
 * using TensorFlow.js or PyTorch bindings.
 */
export class LSTMAutoencoder {
  private sequenceLength = 10;
  private hiddenSize = 32;
  private learningRate = 0.01;

  // Simplified weights (in production, use proper LSTM cells)
  private encoderWeights: number[][] = [];
  private decoderWeights: number[][] = [];
  private meanReconstruction Error = 0;

  constructor(private featureDimensions: number) {
    this.initializeWeights();
  }

  /**
   * Initialize random weights
   */
  private initializeWeights(): void {
    // Encoder: featureDimensions -> hiddenSize
    this.encoderWeights = Array.from({ length: this.hiddenSize }, () =>
      Array.from({ length: this.featureDimensions }, () =>
        (Math.random() - 0.5) * 0.1
      )
    );

    // Decoder: hiddenSize -> featureDimensions
    this.decoderWeights = Array.from({ length: this.featureDimensions }, () =>
      Array.from({ length: this.hiddenSize }, () =>
        (Math.random() - 0.5) * 0.1
      )
    );
  }

  /**
   * Train autoencoder on normal sequences
   */
  async train(data: AnomalyPoint[]): Promise<void> {
    const epochs = 50;
    let totalError = 0;
    let count = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      totalError = 0;
      count = 0;

      // Create sequences
      for (let i = 0; i <= data.length - this.sequenceLength; i++) {
        const sequence = data.slice(i, i + this.sequenceLength);
        const error = this.trainStep(sequence);
        totalError += error;
        count++;
      }

      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}, Avg Error: ${(totalError / count).toFixed(6)}`);
      }
    }

    this.meanReconstructionError = totalError / count;
  }

  /**
   * Single training step
   */
  private trainStep(sequence: AnomalyPoint[]): number {
    // Forward pass
    const encoded = this.encode(sequence);
    const decoded = this.decode(encoded, sequence.length);

    // Compute reconstruction error
    let error = 0;
    for (let t = 0; t < sequence.length; t++) {
      for (let f = 0; f < this.featureDimensions; f++) {
        const diff = decoded[t][f] - sequence[t].features[f];
        error += diff * diff;
      }
    }
    error = Math.sqrt(error / (sequence.length * this.featureDimensions));

    // Simplified backpropagation (gradient descent)
    // In production, use proper LSTM backpropagation through time
    this.updateWeights(sequence, decoded, error);

    return error;
  }

  /**
   * Encode sequence to hidden representation
   */
  private encode(sequence: AnomalyPoint[]): number[] {
    // Simplified: average pooling over sequence
    const hidden = new Array(this.hiddenSize).fill(0);

    for (const point of sequence) {
      for (let h = 0; h < this.hiddenSize; h++) {
        let activation = 0;
        for (let f = 0; f < this.featureDimensions; f++) {
          activation += point.features[f] * this.encoderWeights[h][f];
        }
        hidden[h] += Math.tanh(activation);
      }
    }

    // Average
    return hidden.map(h => h / sequence.length);
  }

  /**
   * Decode hidden representation to sequence
   */
  private decode(hidden: number[], length: number): number[][] {
    const decoded: number[][] = [];

    for (let t = 0; t < length; t++) {
      const features: number[] = [];

      for (let f = 0; f < this.featureDimensions; f++) {
        let activation = 0;
        for (let h = 0; h < this.hiddenSize; h++) {
          activation += hidden[h] * this.decoderWeights[f][h];
        }
        features.push(Math.tanh(activation));
      }

      decoded.push(features);
    }

    return decoded;
  }

  /**
   * Update weights using gradient descent
   */
  private updateWeights(
    sequence: AnomalyPoint[],
    decoded: number[][],
    error: number
  ): void {
    // Simplified weight update
    const gradient = error * this.learningRate;

    for (let h = 0; h < this.hiddenSize; h++) {
      for (let f = 0; f < this.featureDimensions; f++) {
        this.encoderWeights[h][f] -= gradient * (Math.random() - 0.5) * 0.01;
      }
    }

    for (let f = 0; f < this.featureDimensions; f++) {
      for (let h = 0; h < this.hiddenSize; h++) {
        this.decoderWeights[f][h] -= gradient * (Math.random() - 0.5) * 0.01;
      }
    }
  }

  /**
   * Predict anomaly score (reconstruction error)
   */
  predict(point: AnomalyPoint): number {
    // For single point, create a pseudo-sequence by repeating
    const sequence = Array(this.sequenceLength).fill(point);

    const encoded = this.encode(sequence);
    const decoded = this.decode(encoded, 1)[0];

    // Compute reconstruction error
    let error = 0;
    for (let f = 0; f < this.featureDimensions; f++) {
      const diff = decoded[f] - point.features[f];
      error += diff * diff;
    }
    error = Math.sqrt(error / this.featureDimensions);

    // Normalize by mean reconstruction error
    const normalizedScore = error / (this.meanReconstructionError || 1);

    return Math.min(1, normalizedScore);
  }
}
