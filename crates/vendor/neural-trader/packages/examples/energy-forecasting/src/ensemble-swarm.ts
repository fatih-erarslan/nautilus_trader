/**
 * Swarm-based ensemble learning for energy forecasting
 * Explores multiple model architectures and hyperparameters in parallel
 */

import { ForecastingModel, createModel } from './models';
import {
  ModelType,
  TimeSeriesPoint,
  EnsembleConfig,
  TrainingResult,
  SwarmExplorationResult,
  ModelPerformance
} from './types';

/**
 * Hyperparameter configuration for model exploration
 */
interface HyperparameterSpace {
  [ModelType.ARIMA]: { alpha: number[]; beta: number[] };
  [ModelType.LSTM]: { lookback: number[] };
  [ModelType.TRANSFORMER]: { sequenceLength: number[] };
  [ModelType.PROPHET]: { seasonalPeriod: number[] };
  [ModelType.ENSEMBLE]: {};
}

/**
 * Ensemble swarm coordinator
 * Manages parallel model training and adaptive model selection
 */
export class EnsembleSwarm {
  private models: Map<string, ForecastingModel> = new Map();
  private modelPerformance: Map<string, ModelPerformance> = new Map();
  private horizonWeights: Map<number, Map<ModelType, number>> = new Map();
  private config: EnsembleConfig;

  constructor(config: EnsembleConfig) {
    this.config = {
      adaptiveLearningRate: 0.05,
      retrainFrequency: 100,
      minCalibrationSamples: 50,
      ...config
    };
  }

  /**
   * Explore hyperparameter space using swarm intelligence
   */
  async exploreHyperparameters(
    modelType: ModelType,
    data: TimeSeriesPoint[],
    validationData: TimeSeriesPoint[]
  ): Promise<SwarmExplorationResult> {
    const startTime = Date.now();
    const hyperparameterSpace = this.getHyperparameterSpace(modelType);
    const results: TrainingResult[] = [];

    // Generate candidate configurations
    const candidates = this.generateCandidates(modelType, hyperparameterSpace);

    // Train and evaluate each candidate in parallel
    const trainingPromises = candidates.map(async (hyperparams) => {
      const model = createModel(modelType, hyperparams);
      const trainStart = Date.now();

      try {
        await model.train(data);
        const performance = await this.evaluateModel(model, validationData);

        const result: TrainingResult = {
          modelType,
          trainDuration: Date.now() - trainStart,
          samples: data.length,
          performance,
          hyperparameters: hyperparams
        };

        return result;
      } catch (error) {
        console.warn(`Failed to train ${modelType} with hyperparams:`, hyperparams, error);
        return null;
      }
    });

    const allResults = (await Promise.all(trainingPromises)).filter(
      (r): r is TrainingResult => r !== null
    );

    results.push(...allResults);

    // Find best configuration
    const bestResult = results.reduce((best, current) =>
      current.performance.mape < best.performance.mape ? current : best
    );

    const explorationTime = Date.now() - startTime;

    return {
      bestModel: bestResult.modelType,
      bestHyperparameters: bestResult.hyperparameters || {},
      allResults: results,
      explorationTime,
      convergenceMetric: bestResult.performance.mape
    };
  }

  /**
   * Train ensemble with multiple model types
   */
  async trainEnsemble(
    trainingData: TimeSeriesPoint[],
    validationData: TimeSeriesPoint[]
  ): Promise<void> {
    const explorationPromises = this.config.models.map(async (modelType) => {
      console.info(`Exploring ${modelType} hyperparameters...`);
      const result = await this.exploreHyperparameters(
        modelType,
        trainingData,
        validationData
      );

      // Train best model with optimal hyperparameters
      const model = createModel(result.bestModel, result.bestHyperparameters);
      await model.train(trainingData);

      const modelKey = this.getModelKey(modelType, result.bestHyperparameters);
      this.models.set(modelKey, model);
      this.modelPerformance.set(modelKey, result.allResults[0].performance);

      console.info(
        `${modelType} trained with MAPE: ${result.convergenceMetric.toFixed(2)}%`
      );

      return result;
    });

    await Promise.all(explorationPromises);

    // Initialize horizon-based weights
    this.initializeHorizonWeights();
  }

  /**
   * Predict using adaptive ensemble
   */
  async predict(horizon: number): Promise<{ predictions: number[]; selectedModel: string }> {
    if (this.models.size === 0) {
      throw new Error('Ensemble not trained');
    }

    // Get weights for this horizon
    const weights = this.getHorizonWeights(horizon);

    // Get predictions from all models
    const modelPredictions = new Map<string, number[]>();

    for (const [modelKey, model] of this.models.entries()) {
      try {
        const predictions = await model.predict(horizon);
        modelPredictions.set(modelKey, predictions);
      } catch (error) {
        console.warn(`Failed to get predictions from ${modelKey}:`, error);
      }
    }

    if (modelPredictions.size === 0) {
      throw new Error('No models produced valid predictions');
    }

    // Weighted ensemble prediction
    const ensemblePredictions: number[] = Array(horizon).fill(0);

    for (const [modelKey, predictions] of modelPredictions.entries()) {
      const modelType = this.extractModelType(modelKey);
      const weight = weights.get(modelType) || 0;

      for (let i = 0; i < horizon; i++) {
        ensemblePredictions[i] += predictions[i] * weight;
      }
    }

    // Find best performing model for this horizon
    const bestModelEntry = Array.from(this.modelPerformance.entries()).reduce((best, current) =>
      current[1].mape < best[1].mape ? current : best
    );

    return {
      predictions: ensemblePredictions,
      selectedModel: bestModelEntry[0]
    };
  }

  /**
   * Update ensemble with new observations (adaptive learning)
   */
  async updateWithObservation(
    predictions: Map<string, number>,
    actual: number,
    horizon: number
  ): Promise<void> {
    const learningRate = this.config.adaptiveLearningRate || 0.05;

    // Update weights based on prediction errors
    for (const [modelKey, prediction] of predictions.entries()) {
      const error = Math.abs(prediction - actual);
      const modelType = this.extractModelType(modelKey);

      // Get current performance
      const performance = this.modelPerformance.get(modelKey);
      if (!performance) continue;

      // Update MAPE
      const newMape = performance.mape * 0.95 + (error / Math.abs(actual)) * 100 * 0.05;
      performance.mape = newMape;
      performance.lastUpdated = Date.now();

      this.modelPerformance.set(modelKey, performance);

      // Update horizon weights
      const horizonWeights = this.horizonWeights.get(horizon);
      if (horizonWeights) {
        const currentWeight = horizonWeights.get(modelType) || 0;
        const reward = 1 / (1 + error); // Higher reward for lower error
        const newWeight = currentWeight + learningRate * (reward - currentWeight);
        horizonWeights.set(modelType, newWeight);
      }
    }

    // Normalize weights
    this.normalizeHorizonWeights(horizon);
  }

  /**
   * Get ensemble statistics
   */
  getEnsembleStats(): {
    modelCount: number;
    performances: Array<{ model: string; performance: ModelPerformance }>;
    horizonWeights: Map<number, Map<ModelType, number>>;
  } {
    const performances = Array.from(this.modelPerformance.entries()).map(
      ([model, performance]) => ({
        model,
        performance
      })
    );

    return {
      modelCount: this.models.size,
      performances,
      horizonWeights: this.horizonWeights
    };
  }

  /**
   * Get hyperparameter space for model type
   */
  private getHyperparameterSpace(modelType: ModelType): any {
    const space: HyperparameterSpace = {
      [ModelType.ARIMA]: {
        alpha: [0.1, 0.2, 0.3, 0.4, 0.5],
        beta: [0.05, 0.1, 0.15, 0.2]
      },
      [ModelType.LSTM]: {
        lookback: [12, 24, 48, 72, 96]
      },
      [ModelType.TRANSFORMER]: {
        sequenceLength: [24, 48, 72, 96, 168]
      },
      [ModelType.PROPHET]: {
        seasonalPeriod: [24, 48, 168, 336] // hourly, 2-day, weekly, 2-weekly
      },
      [ModelType.ENSEMBLE]: {}
    };

    return space[modelType];
  }

  /**
   * Generate candidate hyperparameter configurations
   */
  private generateCandidates(modelType: ModelType, space: any): Array<Record<string, any>> {
    const candidates: Array<Record<string, any>> = [];

    switch (modelType) {
      case ModelType.ARIMA:
        for (const alpha of space.alpha) {
          for (const beta of space.beta) {
            candidates.push({ alpha, beta });
          }
        }
        break;
      case ModelType.LSTM:
        for (const lookback of space.lookback) {
          candidates.push({ lookback });
        }
        break;
      case ModelType.TRANSFORMER:
        for (const sequenceLength of space.sequenceLength) {
          candidates.push({ sequenceLength });
        }
        break;
      case ModelType.PROPHET:
        for (const seasonalPeriod of space.seasonalPeriod) {
          candidates.push({ seasonalPeriod });
        }
        break;
    }

    return candidates;
  }

  /**
   * Evaluate model performance on validation data
   */
  private async evaluateModel(
    model: ForecastingModel,
    validationData: TimeSeriesPoint[]
  ): Promise<ModelPerformance> {
    const predictions = await model.predict(validationData.length);
    const actuals = validationData.map(d => d.value);

    let sumApe = 0;
    let sumAe = 0;
    let sumSe = 0;

    for (let i = 0; i < predictions.length; i++) {
      const error = Math.abs(predictions[i] - actuals[i]);
      const ape = (error / Math.abs(actuals[i])) * 100;
      sumApe += ape;
      sumAe += error;
      sumSe += error * error;
    }

    const mape = sumApe / predictions.length;
    const mae = sumAe / predictions.length;
    const rmse = Math.sqrt(sumSe / predictions.length);

    return {
      modelName: model.modelType,
      mape,
      mae,
      rmse,
      coverage: 0.9, // Will be updated with conformal prediction
      intervalWidth: 0,
      lastUpdated: Date.now()
    };
  }

  /**
   * Initialize horizon-based weights
   */
  private initializeHorizonWeights(): void {
    const horizons = [1, 6, 12, 24, 48, 72, 168]; // Various forecast horizons

    for (const horizon of horizons) {
      const weights = new Map<ModelType, number>();
      const numModels = this.config.models.length;

      // Equal initial weights
      for (const modelType of this.config.models) {
        weights.set(modelType, 1 / numModels);
      }

      this.horizonWeights.set(horizon, weights);
    }
  }

  /**
   * Get weights for a specific horizon (interpolate if needed)
   */
  private getHorizonWeights(horizon: number): Map<ModelType, number> {
    // Check if exact horizon exists
    if (this.horizonWeights.has(horizon)) {
      return this.horizonWeights.get(horizon)!;
    }

    // Find nearest horizons for interpolation
    const sortedHorizons = Array.from(this.horizonWeights.keys()).sort((a, b) => a - b);
    const lowerHorizon = sortedHorizons.reverse().find(h => h <= horizon);
    const upperHorizon = sortedHorizons.find(h => h >= horizon);

    if (!lowerHorizon || !upperHorizon || lowerHorizon === upperHorizon) {
      // Return default equal weights
      const weights = new Map<ModelType, number>();
      const numModels = this.config.models.length;
      for (const modelType of this.config.models) {
        weights.set(modelType, 1 / numModels);
      }
      return weights;
    }

    // Linear interpolation
    const ratio = (horizon - lowerHorizon) / (upperHorizon - lowerHorizon);
    const lowerWeights = this.horizonWeights.get(lowerHorizon)!;
    const upperWeights = this.horizonWeights.get(upperHorizon)!;

    const interpolatedWeights = new Map<ModelType, number>();
    for (const modelType of this.config.models) {
      const lower = lowerWeights.get(modelType) || 0;
      const upper = upperWeights.get(modelType) || 0;
      interpolatedWeights.set(modelType, lower * (1 - ratio) + upper * ratio);
    }

    return interpolatedWeights;
  }

  /**
   * Normalize weights for a horizon to sum to 1
   */
  private normalizeHorizonWeights(horizon: number): void {
    const weights = this.horizonWeights.get(horizon);
    if (!weights) return;

    const sum = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    if (sum === 0) return;

    for (const [modelType, weight] of weights.entries()) {
      weights.set(modelType, weight / sum);
    }
  }

  /**
   * Generate model key
   */
  private getModelKey(modelType: ModelType, hyperparams: Record<string, any>): string {
    const paramStr = Object.entries(hyperparams)
      .map(([k, v]) => `${k}=${v}`)
      .join(',');
    return `${modelType}:${paramStr}`;
  }

  /**
   * Extract model type from model key
   */
  private extractModelType(modelKey: string): ModelType {
    const type = modelKey.split(':')[0];
    return type as ModelType;
  }
}
