// Type definitions for @neural-trader/neural
import type {
  ModelConfig,
  TrainingConfig,
  TrainingMetrics,
  PredictionResult
} from '@neural-trader/core';

export { ModelConfig, TrainingConfig, TrainingMetrics, PredictionResult };

export class NeuralModel {
  constructor(config: ModelConfig);
  train(
    data: number[],
    targets: number[],
    trainingConfig: TrainingConfig
  ): Promise<TrainingMetrics[]>;
  predict(inputData: number[]): Promise<PredictionResult>;
  save(path: string): Promise<string>;
  load(path: string): Promise<void>;
  getInfo(): Promise<string>;
}

export class BatchPredictor {
  constructor();
  addModel(model: NeuralModel): Promise<number>;
  predictBatch(inputs: number[][]): Promise<PredictionResult[]>;
}

export function listModelTypes(): string[];
