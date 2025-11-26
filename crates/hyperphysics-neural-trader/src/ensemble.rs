//! Ensemble prediction for combining multiple neural models.

use crate::adapter::NeuralFeatures;
use crate::config::{EnsembleMethod, NeuralBridgeConfig, NeuralModelType};
use crate::error::{NeuralBridgeError, Result};
use crate::forecast::{ForecastResult, NeuralForecastEngine};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

/// Ensemble predictor combining multiple neural models
pub struct EnsemblePredictor {
    /// Configuration
    config: NeuralBridgeConfig,
    /// Individual forecast engines per model type
    engines: HashMap<NeuralModelType, NeuralForecastEngine>,
    /// Model weights for weighted averaging
    model_weights: Arc<RwLock<HashMap<NeuralModelType, f64>>>,
    /// Historical performance tracking
    performance_history: Arc<RwLock<PerformanceHistory>>,
}

/// Historical performance metrics for adaptive weighting
#[derive(Debug, Clone, Default)]
struct PerformanceHistory {
    /// Recent prediction errors by model
    recent_errors: HashMap<NeuralModelType, Vec<f64>>,
    /// Moving average of accuracy
    accuracy_ema: HashMap<NeuralModelType, f64>,
    /// Total predictions per model
    prediction_counts: HashMap<NeuralModelType, usize>,
}

impl EnsemblePredictor {
    /// Create a new ensemble predictor
    pub fn new(config: NeuralBridgeConfig) -> Self {
        let mut engines = HashMap::new();
        let mut model_weights = HashMap::new();

        // Initialize engines for each model in ensemble
        let num_models = config.ensemble_models.len();
        let equal_weight = 1.0 / num_models.max(1) as f64;

        for model_type in &config.ensemble_models {
            // Create engine with single-model config
            let single_config = NeuralBridgeConfig {
                ensemble_models: vec![*model_type],
                ensemble_method: EnsembleMethod::Single,
                ..config.clone()
            };
            engines.insert(*model_type, NeuralForecastEngine::new(single_config));
            model_weights.insert(*model_type, equal_weight);
        }

        Self {
            config,
            engines,
            model_weights: Arc::new(RwLock::new(model_weights)),
            performance_history: Arc::new(RwLock::new(PerformanceHistory::default())),
        }
    }

    /// Generate ensemble forecast by combining multiple model predictions
    pub async fn predict(&self, features: &NeuralFeatures) -> Result<ForecastResult> {
        if self.engines.is_empty() {
            return Err(NeuralBridgeError::Configuration(
                "No models configured for ensemble".to_string(),
            ));
        }

        // If only one model or Single method, just use first engine
        if self.engines.len() == 1 || self.config.ensemble_method == EnsembleMethod::Single {
            let (model_type, engine) = self.engines.iter().next().unwrap();
            return engine.forecast(features).await;
        }

        debug!(
            method = ?self.config.ensemble_method,
            num_models = self.engines.len(),
            "Running ensemble prediction"
        );

        // Collect predictions from all models
        let mut model_forecasts: Vec<(NeuralModelType, ForecastResult)> = Vec::new();

        for (model_type, engine) in &self.engines {
            match engine.forecast(features).await {
                Ok(forecast) => {
                    trace!(model = %model_type, "Model forecast successful");
                    model_forecasts.push((*model_type, forecast));
                }
                Err(e) => {
                    debug!(model = %model_type, error = %e, "Model forecast failed, skipping");
                    continue;
                }
            }
        }

        if model_forecasts.is_empty() {
            return Err(NeuralBridgeError::EnsembleAggregation(
                "All models failed to produce forecasts".to_string(),
            ));
        }

        // Aggregate based on method
        let ensemble_result = match self.config.ensemble_method {
            EnsembleMethod::Single => model_forecasts.into_iter().next().unwrap().1,
            EnsembleMethod::SimpleAverage => self.simple_average(model_forecasts).await,
            EnsembleMethod::WeightedAverage => self.weighted_average(model_forecasts).await,
            EnsembleMethod::Median => self.median_ensemble(model_forecasts).await,
            EnsembleMethod::StackedGeneralization => {
                self.stacked_generalization(model_forecasts, features).await
            }
        };

        Ok(ensemble_result)
    }

    /// Simple average of all model predictions
    async fn simple_average(
        &self,
        forecasts: Vec<(NeuralModelType, ForecastResult)>,
    ) -> ForecastResult {
        let n = forecasts.len() as f64;
        let horizon = forecasts[0].1.predictions.len();

        let mut avg_predictions = vec![0.0; horizon];
        let mut avg_lower = vec![0.0; horizon];
        let mut avg_upper = vec![0.0; horizon];
        let mut avg_variance = vec![0.0; horizon];

        for (_, forecast) in &forecasts {
            for h in 0..horizon {
                avg_predictions[h] += forecast.predictions.get(h).copied().unwrap_or(0.0) / n;
                avg_lower[h] += forecast.lower_bound.get(h).copied().unwrap_or(0.0) / n;
                avg_upper[h] += forecast.upper_bound.get(h).copied().unwrap_or(0.0) / n;
                avg_variance[h] += forecast.variance.get(h).copied().unwrap_or(0.0) / n;
            }
        }

        ForecastResult {
            predictions: avg_predictions,
            lower_bound: avg_lower,
            upper_bound: avg_upper,
            variance: avg_variance,
            model_type: NeuralModelType::NHITS, // Ensemble marker
            confidence_level: forecasts[0].1.confidence_level,
            feature_importance: None,
            horizon,
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        }
    }

    /// Weighted average based on historical performance
    async fn weighted_average(
        &self,
        forecasts: Vec<(NeuralModelType, ForecastResult)>,
    ) -> ForecastResult {
        let weights = self.model_weights.read().await;
        let horizon = forecasts[0].1.predictions.len();

        // Normalize weights for available models
        let total_weight: f64 = forecasts.iter()
            .map(|(m, _)| weights.get(m).copied().unwrap_or(1.0))
            .sum();

        let mut weighted_predictions = vec![0.0; horizon];
        let mut weighted_lower = vec![0.0; horizon];
        let mut weighted_upper = vec![0.0; horizon];
        let mut weighted_variance = vec![0.0; horizon];

        for (model_type, forecast) in &forecasts {
            let w = weights.get(model_type).copied().unwrap_or(1.0) / total_weight.max(1e-8);

            for h in 0..horizon {
                weighted_predictions[h] += forecast.predictions.get(h).copied().unwrap_or(0.0) * w;
                weighted_lower[h] += forecast.lower_bound.get(h).copied().unwrap_or(0.0) * w;
                weighted_upper[h] += forecast.upper_bound.get(h).copied().unwrap_or(0.0) * w;
                weighted_variance[h] += forecast.variance.get(h).copied().unwrap_or(0.0) * w;
            }
        }

        ForecastResult {
            predictions: weighted_predictions,
            lower_bound: weighted_lower,
            upper_bound: weighted_upper,
            variance: weighted_variance,
            model_type: NeuralModelType::NHITS,
            confidence_level: forecasts[0].1.confidence_level,
            feature_importance: None,
            horizon,
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        }
    }

    /// Median ensemble (robust to outliers)
    async fn median_ensemble(
        &self,
        forecasts: Vec<(NeuralModelType, ForecastResult)>,
    ) -> ForecastResult {
        let horizon = forecasts[0].1.predictions.len();

        let mut median_predictions = Vec::with_capacity(horizon);
        let mut median_lower = Vec::with_capacity(horizon);
        let mut median_upper = Vec::with_capacity(horizon);
        let mut median_variance = Vec::with_capacity(horizon);

        for h in 0..horizon {
            // Collect all predictions for this horizon
            let mut preds: Vec<f64> = forecasts.iter()
                .filter_map(|(_, f)| f.predictions.get(h).copied())
                .collect();
            let mut lowers: Vec<f64> = forecasts.iter()
                .filter_map(|(_, f)| f.lower_bound.get(h).copied())
                .collect();
            let mut uppers: Vec<f64> = forecasts.iter()
                .filter_map(|(_, f)| f.upper_bound.get(h).copied())
                .collect();
            let mut vars: Vec<f64> = forecasts.iter()
                .filter_map(|(_, f)| f.variance.get(h).copied())
                .collect();

            median_predictions.push(Self::median(&mut preds));
            median_lower.push(Self::median(&mut lowers));
            median_upper.push(Self::median(&mut uppers));
            median_variance.push(Self::median(&mut vars));
        }

        ForecastResult {
            predictions: median_predictions,
            lower_bound: median_lower,
            upper_bound: median_upper,
            variance: median_variance,
            model_type: NeuralModelType::NHITS,
            confidence_level: forecasts[0].1.confidence_level,
            feature_importance: None,
            horizon,
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        }
    }

    /// Stacked generalization (meta-learner)
    async fn stacked_generalization(
        &self,
        forecasts: Vec<(NeuralModelType, ForecastResult)>,
        features: &NeuralFeatures,
    ) -> ForecastResult {
        // In full implementation, this would train a meta-model
        // For standalone mode, use weighted average with quality-based weights
        let horizon = forecasts[0].1.predictions.len();

        // Calculate quality-based weights from forecast uncertainty
        let mut quality_weights: HashMap<NeuralModelType, f64> = HashMap::new();
        let mut total_quality = 0.0;

        for (model_type, forecast) in &forecasts {
            let quality = forecast.quality_score();
            quality_weights.insert(*model_type, quality);
            total_quality += quality;
        }

        // Normalize weights
        for (_, weight) in quality_weights.iter_mut() {
            *weight /= total_quality.max(1e-8);
        }

        // Apply quality weights
        let mut weighted_predictions = vec![0.0; horizon];
        let mut weighted_lower = vec![0.0; horizon];
        let mut weighted_upper = vec![0.0; horizon];
        let mut weighted_variance = vec![0.0; horizon];

        for (model_type, forecast) in &forecasts {
            let w = quality_weights.get(model_type).copied().unwrap_or(0.0);

            for h in 0..horizon {
                weighted_predictions[h] += forecast.predictions.get(h).copied().unwrap_or(0.0) * w;
                weighted_lower[h] += forecast.lower_bound.get(h).copied().unwrap_or(0.0) * w;
                weighted_upper[h] += forecast.upper_bound.get(h).copied().unwrap_or(0.0) * w;
                weighted_variance[h] += forecast.variance.get(h).copied().unwrap_or(0.0) * w;
            }
        }

        ForecastResult {
            predictions: weighted_predictions,
            lower_bound: weighted_lower,
            upper_bound: weighted_upper,
            variance: weighted_variance,
            model_type: NeuralModelType::NHITS,
            confidence_level: forecasts[0].1.confidence_level,
            feature_importance: None,
            horizon,
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        }
    }

    /// Calculate median of a slice
    fn median(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Update model weights based on prediction error
    pub async fn update_weights(&self, model_type: NeuralModelType, error: f64) {
        let mut history = self.performance_history.write().await;

        // Track recent errors
        let errors = history.recent_errors
            .entry(model_type)
            .or_insert_with(Vec::new);
        errors.push(error.abs());

        // Keep last 100 errors
        if errors.len() > 100 {
            errors.remove(0);
        }

        // Update EMA of accuracy (lower error = higher accuracy)
        let alpha = 0.1;
        let current_accuracy = 1.0 / (1.0 + error.abs());
        let ema = history.accuracy_ema
            .entry(model_type)
            .or_insert(current_accuracy);
        *ema = alpha * current_accuracy + (1.0 - alpha) * *ema;

        // Increment prediction count
        *history.prediction_counts.entry(model_type).or_insert(0) += 1;

        // Recalculate weights based on EMA accuracy
        drop(history);
        self.recalculate_weights().await;
    }

    /// Recalculate model weights from performance history
    async fn recalculate_weights(&self) {
        let history = self.performance_history.read().await;

        let mut new_weights = HashMap::new();
        let mut total_accuracy = 0.0;

        for model_type in self.config.ensemble_models.iter() {
            let accuracy = history.accuracy_ema.get(model_type).copied().unwrap_or(0.5);
            new_weights.insert(*model_type, accuracy);
            total_accuracy += accuracy;
        }

        // Normalize
        for (_, weight) in new_weights.iter_mut() {
            *weight /= total_accuracy.max(1e-8);
        }

        drop(history);

        // Apply new weights
        let mut weights = self.model_weights.write().await;
        *weights = new_weights;
    }

    /// Get current model weights
    pub async fn get_weights(&self) -> HashMap<NeuralModelType, f64> {
        self.model_weights.read().await.clone()
    }

    /// Get ensemble configuration
    pub fn config(&self) -> &NeuralBridgeConfig {
        &self.config
    }

    /// Get number of models in ensemble
    pub fn model_count(&self) -> usize {
        self.engines.len()
    }
}

impl Clone for EnsemblePredictor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            engines: self.engines.clone(),
            model_weights: Arc::clone(&self.model_weights),
            performance_history: Arc::clone(&self.performance_history),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_features(prices: Vec<f64>) -> NeuralFeatures {
        let n = prices.len();
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        NeuralFeatures {
            prices: prices.clone(),
            returns,
            volatility: vec![0.02; n],
            spreads: vec![0.001; n],
            vwaps: prices.iter().map(|p| p * 0.999).collect(),
            feature_matrix: Array2::zeros((n, 5)),
            targets: None,
            timestamps: (0..n).map(|i| i as f64 * 1000.0).collect(),
        }
    }

    #[tokio::test]
    async fn test_ensemble_creation() {
        let config = NeuralBridgeConfig {
            ensemble_models: vec![NeuralModelType::NHITS, NeuralModelType::LSTMAttention],
            ensemble_method: EnsembleMethod::WeightedAverage,
            ..Default::default()
        };

        let ensemble = EnsemblePredictor::new(config);
        assert_eq!(ensemble.model_count(), 2);
    }

    #[tokio::test]
    async fn test_simple_average_ensemble() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 10,
            forecast_horizon: 4,
            ensemble_models: vec![NeuralModelType::NHITS, NeuralModelType::LSTMAttention],
            ensemble_method: EnsembleMethod::SimpleAverage,
            ..Default::default()
        };

        let ensemble = EnsemblePredictor::new(config);
        let prices: Vec<f64> = (0..24).map(|i| 100.0 + i as f64 * 0.5).collect();
        let features = make_test_features(prices);

        let result = ensemble.predict(&features).await.unwrap();
        assert_eq!(result.predictions.len(), 4);
    }

    #[tokio::test]
    async fn test_weighted_average_ensemble() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 10,
            forecast_horizon: 4,
            ensemble_models: vec![NeuralModelType::NHITS, NeuralModelType::Transformer],
            ensemble_method: EnsembleMethod::WeightedAverage,
            ..Default::default()
        };

        let ensemble = EnsemblePredictor::new(config);
        let prices: Vec<f64> = (0..24).map(|i| 100.0 + i as f64 * 0.2).collect();
        let features = make_test_features(prices);

        let result = ensemble.predict(&features).await.unwrap();
        assert_eq!(result.predictions.len(), 4);
        assert!(result.predictions[0] > 0.0);
    }

    #[tokio::test]
    async fn test_median_ensemble() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 10,
            forecast_horizon: 4,
            ensemble_models: vec![
                NeuralModelType::NHITS,
                NeuralModelType::LSTMAttention,
                NeuralModelType::Transformer,
            ],
            ensemble_method: EnsembleMethod::Median,
            ..Default::default()
        };

        let ensemble = EnsemblePredictor::new(config);
        let prices: Vec<f64> = (0..24).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let features = make_test_features(prices);

        let result = ensemble.predict(&features).await.unwrap();
        assert_eq!(result.predictions.len(), 4);
    }

    #[tokio::test]
    async fn test_weight_updates() {
        let config = NeuralBridgeConfig {
            ensemble_models: vec![NeuralModelType::NHITS, NeuralModelType::LSTMAttention],
            ..Default::default()
        };

        let ensemble = EnsemblePredictor::new(config);

        // Simulate better performance for NHITS
        for _ in 0..10 {
            ensemble.update_weights(NeuralModelType::NHITS, 0.01).await;
            ensemble.update_weights(NeuralModelType::LSTMAttention, 0.05).await;
        }

        let weights = ensemble.get_weights().await;
        let nhits_weight = weights.get(&NeuralModelType::NHITS).unwrap();
        let lstm_weight = weights.get(&NeuralModelType::LSTMAttention).unwrap();

        // NHITS should have higher weight due to lower error
        assert!(nhits_weight > lstm_weight);
    }

    #[test]
    fn test_median_calculation() {
        let mut odd = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(EnsemblePredictor::median(&mut odd), 3.0);

        let mut even = vec![1.0, 4.0, 3.0, 2.0];
        assert_eq!(EnsemblePredictor::median(&mut even), 2.5);

        let mut empty: Vec<f64> = vec![];
        assert_eq!(EnsemblePredictor::median(&mut empty), 0.0);
    }

    #[tokio::test]
    async fn test_single_model_mode() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 10,
            forecast_horizon: 4,
            ensemble_models: vec![NeuralModelType::NHITS],
            ensemble_method: EnsembleMethod::Single,
            ..Default::default()
        };

        let ensemble = EnsemblePredictor::new(config);
        let prices: Vec<f64> = (0..24).map(|i| 100.0 + i as f64).collect();
        let features = make_test_features(prices);

        let result = ensemble.predict(&features).await.unwrap();
        assert_eq!(result.model_type, NeuralModelType::NHITS);
    }
}
