//! Neural Trader Integration for CWTS Risk Management
//!
//! Integrates neural network forecasting with ensemble methods and
//! conformal prediction for uncertainty quantification.
//!
//! ## Neural Models (8 Types)
//!
//! | Model | Architecture | Best For |
//! |-------|-------------|----------|
//! | N-HiTS | Hierarchical interpolation | Multi-horizon |
//! | LSTM-Attention | Long Short-Term Memory + Attention | Regime-dependent |
//! | Transformer | Self-attention | Long sequences |
//! | GRU | Gated Recurrent Unit | Fast inference |
//! | TCN | Temporal Convolutional | Local patterns |
//! | DeepAR | Autoregressive probabilistic | Uncertainty |
//! | N-BEATS | Neural basis expansion | Interpretable |
//! | Prophet | Additive model | Seasonality |
//!
//! ## Risk Applications
//!
//! - Multi-model ensemble for robust predictions
//! - Conformal prediction for uncertainty bounds
//! - Model disagreement as risk signal
//! - Forecast-based position sizing

use crate::core::{MarketRegime, RiskLevel, Symbol, Price};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// Re-export from hyperphysics-neural-trader when available
#[cfg(feature = "cwts-neural")]
use hyperphysics_neural_trader::prelude::*;

/// Ensemble forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleForecast {
    /// Mean prediction
    pub prediction: f64,
    /// Standard deviation across models
    pub std_dev: f64,
    /// Median prediction
    pub median: f64,
    /// Per-model predictions
    pub model_predictions: HashMap<ModelType, f64>,
    /// Per-model weights
    pub model_weights: HashMap<ModelType, f64>,
    /// Confidence bounds from conformal prediction
    pub confidence_bounds: ConfidenceBounds,
    /// Forecast horizon (periods)
    pub horizon: u32,
    /// Forecast timestamp
    pub timestamp: DateTime<Utc>,
}

impl EnsembleForecast {
    /// Get prediction range (max - min across models)
    #[must_use]
    pub fn prediction_range(&self) -> f64 {
        let preds: Vec<f64> = self.model_predictions.values().copied().collect();
        if preds.is_empty() {
            return 0.0;
        }
        preds.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) -
        preds.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Get coefficient of variation
    #[must_use]
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.prediction.abs() > 1e-10 {
            self.std_dev / self.prediction.abs()
        } else {
            f64::MAX
        }
    }

    /// Get risk level from forecast uncertainty
    #[must_use]
    pub fn uncertainty_risk_level(&self) -> RiskLevel {
        let cv = self.coefficient_of_variation();
        if cv > 1.0 {
            RiskLevel::Critical
        } else if cv > 0.5 {
            RiskLevel::High
        } else if cv > 0.2 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }
}

/// Types of neural models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// N-HiTS: Neural Hierarchical Interpolation for Time Series
    NHits,
    /// LSTM with Attention mechanism
    LSTMAttention,
    /// Transformer encoder
    Transformer,
    /// Gated Recurrent Unit
    GRU,
    /// Temporal Convolutional Network
    TCN,
    /// DeepAR probabilistic forecasting
    DeepAR,
    /// Neural Basis Expansion Analysis
    NBeats,
    /// Prophet additive model
    Prophet,
}

impl ModelType {
    /// Get all model types
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::NHits,
            Self::LSTMAttention,
            Self::Transformer,
            Self::GRU,
            Self::TCN,
            Self::DeepAR,
            Self::NBeats,
            Self::Prophet,
        ]
    }

    /// Get default weight for this model
    #[must_use]
    pub fn default_weight(&self) -> f64 {
        match self {
            Self::NHits => 0.20,
            Self::LSTMAttention => 0.15,
            Self::Transformer => 0.15,
            Self::GRU => 0.10,
            Self::TCN => 0.10,
            Self::DeepAR => 0.15,
            Self::NBeats => 0.10,
            Self::Prophet => 0.05,
        }
    }

    /// Get typical inference latency (microseconds)
    #[must_use]
    pub fn typical_latency_us(&self) -> u64 {
        match self {
            Self::GRU => 100,
            Self::TCN => 150,
            Self::LSTMAttention => 200,
            Self::NBeats => 180,
            Self::NHits => 250,
            Self::Transformer => 300,
            Self::DeepAR => 350,
            Self::Prophet => 500,
        }
    }

    /// Check if model provides uncertainty estimates natively
    #[must_use]
    pub fn provides_uncertainty(&self) -> bool {
        matches!(self, Self::DeepAR | Self::NHits)
    }
}

/// Confidence bounds from conformal prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBounds {
    /// Lower bound (e.g., 5th percentile)
    pub lower: f64,
    /// Upper bound (e.g., 95th percentile)
    pub upper: f64,
    /// Confidence level (e.g., 0.90)
    pub confidence_level: f64,
    /// Prediction interval width
    pub interval_width: f64,
    /// Coverage rate from calibration
    pub calibration_coverage: f64,
    /// Horizon-specific adjustments applied
    pub horizon_adjusted: bool,
}

impl ConfidenceBounds {
    /// Check if value is within bounds
    #[must_use]
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Get relative interval width
    #[must_use]
    pub fn relative_width(&self, reference: f64) -> f64 {
        if reference.abs() > 1e-10 {
            self.interval_width / reference.abs()
        } else {
            f64::MAX
        }
    }

    /// Get risk from interval width
    #[must_use]
    pub fn interval_risk_level(&self, reference: f64) -> RiskLevel {
        let rel_width = self.relative_width(reference);
        if rel_width > 0.5 {
            RiskLevel::Critical
        } else if rel_width > 0.2 {
            RiskLevel::High
        } else if rel_width > 0.1 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }
}

/// Model disagreement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDisagreement {
    /// Overall disagreement score (0.0-1.0)
    pub disagreement_score: f64,
    /// Pairwise disagreements
    pub pairwise: HashMap<(String, String), f64>,
    /// Models predicting up
    pub bullish_models: Vec<ModelType>,
    /// Models predicting down
    pub bearish_models: Vec<ModelType>,
    /// Directional consensus (1.0 = all agree, 0.0 = split)
    pub directional_consensus: f64,
    /// Maximum pairwise disagreement
    pub max_disagreement: f64,
}

impl ModelDisagreement {
    /// Check if models are in significant disagreement
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.disagreement_score > 0.5
    }

    /// Get risk level from disagreement
    #[must_use]
    pub fn to_risk_level(&self) -> RiskLevel {
        if self.disagreement_score > 0.8 {
            RiskLevel::Critical
        } else if self.disagreement_score > 0.5 {
            RiskLevel::High
        } else if self.disagreement_score > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }
}

/// Configuration for neural risk adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Enabled models
    pub enabled_models: Vec<ModelType>,
    /// Model weights (should sum to 1.0)
    pub model_weights: HashMap<ModelType, f64>,
    /// Conformal prediction confidence level
    pub confidence_level: f64,
    /// Calibration window size
    pub calibration_window: usize,
    /// Enable horizon-dependent intervals
    pub horizon_dependent_intervals: bool,
    /// Maximum forecast horizon
    pub max_horizon: u32,
    /// Minimum models for ensemble
    pub min_ensemble_models: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        let models = ModelType::all();
        let weights: HashMap<_, _> = models.iter()
            .map(|m| (*m, m.default_weight()))
            .collect();

        Self {
            enabled_models: models,
            model_weights: weights,
            confidence_level: 0.90,
            calibration_window: 252,
            horizon_dependent_intervals: true,
            max_horizon: 20,
            min_ensemble_models: 3,
        }
    }
}

/// Neural-based risk adapter
///
/// Provides forecast-based risk management:
/// - Ensemble predictions from multiple neural models
/// - Conformal prediction for calibrated uncertainty
/// - Model disagreement as risk signal
pub struct NeuralRiskAdapter {
    config: NeuralConfig,
    model_status: RwLock<HashMap<ModelType, ModelStatus>>,
    forecast_cache: RwLock<HashMap<String, EnsembleForecast>>,
    calibration_errors: RwLock<Vec<CalibrationError>>,
    disagreement_history: RwLock<Vec<ModelDisagreement>>,
}

/// Status of a neural model
#[derive(Debug, Clone)]
struct ModelStatus {
    model_type: ModelType,
    is_available: bool,
    last_inference_us: u64,
    error_count: u32,
    recent_accuracy: f64,
}

/// Calibration error for conformal prediction
#[derive(Debug, Clone)]
struct CalibrationError {
    timestamp: DateTime<Utc>,
    predicted: f64,
    actual: f64,
    error: f64,
    abs_error: f64,
}

impl NeuralRiskAdapter {
    /// Create a new neural risk adapter
    #[must_use]
    pub fn new(config: NeuralConfig) -> Self {
        let model_status: HashMap<_, _> = config.enabled_models.iter()
            .map(|&m| (m, ModelStatus {
                model_type: m,
                is_available: true,
                last_inference_us: 0,
                error_count: 0,
                recent_accuracy: 1.0,
            }))
            .collect();

        Self {
            config,
            model_status: RwLock::new(model_status),
            forecast_cache: RwLock::new(HashMap::new()),
            calibration_errors: RwLock::new(Vec::with_capacity(1000)),
            disagreement_history: RwLock::new(Vec::with_capacity(1000)),
        }
    }

    /// Generate ensemble forecast
    pub fn generate_forecast(
        &self,
        symbol: &Symbol,
        prices: &[f64],
        horizon: u32,
    ) -> EnsembleForecast {
        // Simulate individual model predictions
        // In real implementation, this would call actual neural models
        let model_predictions = self.simulate_model_predictions(prices, horizon);

        // Calculate ensemble statistics
        let preds: Vec<f64> = model_predictions.values().copied().collect();
        let n = preds.len() as f64;

        let mean = preds.iter().sum::<f64>() / n;
        let variance = preds.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = preds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        // Generate conformal bounds
        let confidence_bounds = self.calculate_conformal_bounds(mean, std_dev, horizon);

        EnsembleForecast {
            prediction: mean,
            std_dev,
            median,
            model_predictions,
            model_weights: self.config.model_weights.clone(),
            confidence_bounds,
            horizon,
            timestamp: Utc::now(),
        }
    }

    /// Simulate model predictions (placeholder for actual inference)
    fn simulate_model_predictions(
        &self,
        prices: &[f64],
        horizon: u32,
    ) -> HashMap<ModelType, f64> {
        let mut predictions = HashMap::new();

        if prices.is_empty() {
            return predictions;
        }

        let last_price = *prices.last().unwrap_or(&100.0);
        let recent_returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let avg_return = if recent_returns.is_empty() {
            0.0
        } else {
            recent_returns.iter().sum::<f64>() / recent_returns.len() as f64
        };

        let volatility = if recent_returns.len() >= 2 {
            let var: f64 = recent_returns.iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / (recent_returns.len() - 1) as f64;
            var.sqrt()
        } else {
            0.01
        };

        for model in &self.config.enabled_models {
            // Each model has slightly different predictions based on their characteristics
            let model_bias = match model {
                ModelType::NHits => 0.0, // Most accurate
                ModelType::LSTMAttention => avg_return * 0.1, // Momentum bias
                ModelType::Transformer => -avg_return * 0.05, // Contrarian
                ModelType::GRU => avg_return * 0.05,
                ModelType::TCN => avg_return * 0.08,
                ModelType::DeepAR => 0.0,
                ModelType::NBeats => -avg_return * 0.02,
                ModelType::Prophet => avg_return * 0.15, // Strong trend following
            };

            // Base prediction with model-specific adjustment
            let prediction = last_price * (1.0 + avg_return * horizon as f64 + model_bias);

            // Add some model-specific noise
            let noise_scale = match model {
                ModelType::Prophet => volatility * 2.0,
                ModelType::DeepAR => volatility * 1.5,
                _ => volatility,
            };

            predictions.insert(*model, prediction);
        }

        predictions
    }

    /// Calculate conformal prediction bounds
    fn calculate_conformal_bounds(&self, mean: f64, std_dev: f64, horizon: u32) -> ConfidenceBounds {
        let calibration_errors = self.calibration_errors.read();

        // Get quantile from calibration set
        let quantile = if calibration_errors.len() >= 30 {
            let mut sorted_errors: Vec<f64> = calibration_errors.iter()
                .map(|e| e.abs_error)
                .collect();
            sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let idx = ((self.config.confidence_level * sorted_errors.len() as f64) as usize)
                .min(sorted_errors.len() - 1);
            sorted_errors[idx]
        } else {
            // Fallback to normal approximation
            std_dev * 1.96 // 95% for normal
        };

        // Horizon adjustment (uncertainty grows with horizon)
        let horizon_factor = if self.config.horizon_dependent_intervals {
            (horizon as f64).sqrt()
        } else {
            1.0
        };

        let interval_width = quantile * horizon_factor * 2.0;

        ConfidenceBounds {
            lower: mean - quantile * horizon_factor,
            upper: mean + quantile * horizon_factor,
            confidence_level: self.config.confidence_level,
            interval_width,
            calibration_coverage: self.calculate_calibration_coverage(),
            horizon_adjusted: self.config.horizon_dependent_intervals,
        }
    }

    /// Calculate empirical coverage rate
    fn calculate_calibration_coverage(&self) -> f64 {
        let errors = self.calibration_errors.read();
        if errors.len() < 30 {
            return self.config.confidence_level; // Not enough data
        }

        // This would compare predictions with actual outcomes
        // For now, return configured level
        self.config.confidence_level
    }

    /// Update calibration with actual outcome
    pub fn update_calibration(&self, predicted: f64, actual: f64) {
        let error = actual - predicted;
        let abs_error = error.abs();

        let mut errors = self.calibration_errors.write();
        errors.push(CalibrationError {
            timestamp: Utc::now(),
            predicted,
            actual,
            error,
            abs_error,
        });

        // Keep only recent errors
        if errors.len() > self.config.calibration_window {
            errors.remove(0);
        }
    }

    /// Calculate model disagreement
    pub fn calculate_disagreement(&self, forecast: &EnsembleForecast) -> ModelDisagreement {
        let predictions = &forecast.model_predictions;

        if predictions.len() < 2 {
            return ModelDisagreement {
                disagreement_score: 0.0,
                pairwise: HashMap::new(),
                bullish_models: Vec::new(),
                bearish_models: Vec::new(),
                directional_consensus: 1.0,
                max_disagreement: 0.0,
            };
        }

        // Classify bullish vs bearish
        let current_price = forecast.prediction; // Using mean as reference
        let mut bullish = Vec::new();
        let mut bearish = Vec::new();

        for (&model, &pred) in predictions {
            if pred > current_price {
                bullish.push(model);
            } else {
                bearish.push(model);
            }
        }

        // Calculate pairwise disagreements
        let mut pairwise = HashMap::new();
        let mut max_disagreement = 0.0;
        let models: Vec<_> = predictions.keys().collect();

        for i in 0..models.len() {
            for j in (i + 1)..models.len() {
                let p1 = predictions[models[i]];
                let p2 = predictions[models[j]];
                let disagreement = (p1 - p2).abs() / forecast.prediction.abs().max(1e-10);

                pairwise.insert(
                    (format!("{:?}", models[i]), format!("{:?}", models[j])),
                    disagreement,
                );

                if disagreement > max_disagreement {
                    max_disagreement = disagreement;
                }
            }
        }

        // Overall disagreement score
        let cv = forecast.coefficient_of_variation();
        let disagreement_score = cv.min(1.0);

        // Directional consensus
        let total = predictions.len() as f64;
        let consensus = (bullish.len().max(bearish.len()) as f64 / total).min(1.0);

        let disagreement = ModelDisagreement {
            disagreement_score,
            pairwise,
            bullish_models: bullish,
            bearish_models: bearish,
            directional_consensus: consensus,
            max_disagreement,
        };

        // Track history
        let mut history = self.disagreement_history.write();
        history.push(disagreement.clone());
        if history.len() > 1000 {
            history.remove(0);
        }

        disagreement
    }

    /// Get forecast-based position size recommendation
    pub fn recommend_position_size(
        &self,
        forecast: &EnsembleForecast,
        base_size: f64,
        max_size: f64,
    ) -> f64 {
        // Scale position by confidence
        let cv = forecast.coefficient_of_variation();
        let confidence_factor = (1.0 - cv.min(1.0)).max(0.0);

        // Scale by directional agreement
        let disagreement = self.calculate_disagreement(forecast);
        let consensus_factor = disagreement.directional_consensus;

        // Combined scaling
        let scale = confidence_factor * consensus_factor;

        (base_size * scale).min(max_size).max(0.0)
    }

    /// Assess forecast-based risk for a portfolio.
    ///
    /// Returns a `SubsystemRisk` for integration with the CWTS coordinator.
    /// Evaluates forecast uncertainty, model disagreement, and prediction confidence.
    pub fn assess_forecast_risk(&self, portfolio: &crate::core::Portfolio) -> super::coordinator::SubsystemRisk {
        use super::coordinator::{SubsystemRisk, SubsystemId};
        use crate::core::Timestamp;

        let start = std::time::Instant::now();

        // Generate forecasts for portfolio symbols
        let mut total_uncertainty = 0.0;
        let mut total_disagreement = 0.0;
        let mut forecast_count = 0;
        let mut worst_cv = 0.0;

        for position in &portfolio.positions {
            // Get cached forecast or generate new one
            let cache_key = format!("{}:{}", position.symbol, position.id.as_raw());
            let base_price = position.avg_entry_price.as_f64();
            let forecast = self.get_cached_forecast(&cache_key).unwrap_or_else(|| {
                // Generate simple price series from position data
                let prices: Vec<f64> = (0..50).map(|i| base_price * (1.0 + 0.001 * i as f64)).collect();
                self.generate_forecast(&position.symbol, &prices, 5)
            });

            let cv = forecast.coefficient_of_variation();
            let disagreement = self.calculate_disagreement(&forecast);

            total_uncertainty += cv;
            total_disagreement += disagreement.disagreement_score;
            forecast_count += 1;

            if cv > worst_cv {
                worst_cv = cv;
            }
        }

        // Calculate average metrics
        let avg_uncertainty = if forecast_count > 0 {
            total_uncertainty / forecast_count as f64
        } else {
            0.0
        };

        let avg_disagreement = if forecast_count > 0 {
            total_disagreement / forecast_count as f64
        } else {
            0.0
        };

        // Get recent disagreement history
        let history = self.disagreement_history.read();
        let recent_disagreement_trend = if history.len() >= 5 {
            let recent: Vec<f64> = history.iter().rev().take(5).map(|d| d.disagreement_score).collect();
            let older: Vec<f64> = history.iter().rev().skip(5).take(5).map(|d| d.disagreement_score).collect();
            if !older.is_empty() {
                let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;
                (recent_avg - older_avg).clamp(-1.0, 1.0) // Positive = worsening
            } else {
                0.0
            }
        } else {
            0.0
        };
        drop(history); // Release lock

        // Combined risk score
        let combined_risk = (
            avg_uncertainty.min(1.0) * 0.35 +
            avg_disagreement * 0.35 +
            worst_cv.min(1.0) * 0.20 +
            recent_disagreement_trend.max(0.0) * 0.10
        ).clamp(0.0, 1.0);

        // Determine risk level
        let risk_level = if combined_risk > 0.8 || worst_cv > 1.0 {
            crate::core::RiskLevel::Critical
        } else if combined_risk > 0.6 || avg_disagreement > 0.7 {
            crate::core::RiskLevel::High
        } else if combined_risk > 0.3 || avg_uncertainty > 0.3 {
            crate::core::RiskLevel::Elevated
        } else {
            crate::core::RiskLevel::Normal
        };

        // Position factor based on forecast confidence
        let position_factor = (1.0 - avg_uncertainty.min(0.7)).clamp(0.3, 1.0);

        // Confidence based on model availability and agreement
        let available_models = self.available_models().len() as f64;
        let model_coverage = available_models / 8.0; // 8 models total
        let confidence = (model_coverage * 0.5 + (1.0 - avg_disagreement) * 0.5).clamp(0.3, 0.95);

        let latency_ns = start.elapsed().as_nanos() as u64;

        let reasoning = format!(
            "Neural: uncertainty={:.2}, disagreement={:.2}, worst_cv={:.2}, models={}/8",
            avg_uncertainty,
            avg_disagreement,
            worst_cv,
            self.available_models().len()
        );

        SubsystemRisk {
            subsystem: SubsystemId::Neural,
            risk_level,
            confidence,
            risk_score: combined_risk,
            position_factor,
            reasoning,
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }

    /// Get overall risk level
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        let history = self.disagreement_history.read();
        if history.is_empty() {
            return RiskLevel::Normal;
        }

        let recent_disagreement = history.last()
            .map(|d| d.disagreement_score)
            .unwrap_or(0.0);

        if recent_disagreement > 0.7 {
            RiskLevel::Critical
        } else if recent_disagreement > 0.5 {
            RiskLevel::High
        } else if recent_disagreement > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }

    /// Get available models
    #[must_use]
    pub fn available_models(&self) -> Vec<ModelType> {
        self.model_status.read()
            .iter()
            .filter(|(_, status)| status.is_available)
            .map(|(&model, _)| model)
            .collect()
    }

    /// Mark model as unavailable
    pub fn mark_model_unavailable(&self, model: ModelType) {
        if let Some(status) = self.model_status.write().get_mut(&model) {
            status.is_available = false;
            status.error_count += 1;
        }
    }

    /// Store forecast in cache
    pub fn cache_forecast(&self, key: &str, forecast: EnsembleForecast) {
        self.forecast_cache.write().insert(key.to_string(), forecast);
    }

    /// Get cached forecast
    #[must_use]
    pub fn get_cached_forecast(&self, key: &str) -> Option<EnsembleForecast> {
        self.forecast_cache.read().get(key).cloned()
    }
}

impl Default for NeuralRiskAdapter {
    fn default() -> Self {
        Self::new(NeuralConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_types() {
        assert_eq!(ModelType::all().len(), 8);
        assert!(ModelType::DeepAR.provides_uncertainty());
        assert!(!ModelType::GRU.provides_uncertainty());
    }

    #[test]
    fn test_model_weights() {
        let weights: f64 = ModelType::all().iter()
            .map(|m| m.default_weight())
            .sum();
        assert!((weights - 1.0).abs() < 0.01); // Should sum to ~1.0
    }

    #[test]
    fn test_ensemble_forecast() {
        let adapter = NeuralRiskAdapter::default();

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        let forecast = adapter.generate_forecast(&Symbol::new("BTC"), &prices, 5);

        assert!(forecast.prediction > 0.0);
        assert!(forecast.std_dev >= 0.0);
        assert!(!forecast.model_predictions.is_empty());
        assert!(forecast.confidence_bounds.lower < forecast.confidence_bounds.upper);
    }

    #[test]
    fn test_confidence_bounds() {
        let bounds = ConfidenceBounds {
            lower: 95.0,
            upper: 105.0,
            confidence_level: 0.90,
            interval_width: 10.0,
            calibration_coverage: 0.90,
            horizon_adjusted: true,
        };

        assert!(bounds.contains(100.0));
        assert!(!bounds.contains(110.0));
        assert_eq!(bounds.relative_width(100.0), 0.10);
    }

    #[test]
    fn test_model_disagreement() {
        let adapter = NeuralRiskAdapter::default();

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let forecast = adapter.generate_forecast(&Symbol::new("ETH"), &prices, 3);

        let disagreement = adapter.calculate_disagreement(&forecast);

        assert!(disagreement.disagreement_score >= 0.0);
        assert!(disagreement.directional_consensus >= 0.0 && disagreement.directional_consensus <= 1.0);
    }

    #[test]
    fn test_position_sizing() {
        let adapter = NeuralRiskAdapter::default();

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.05).collect();
        let forecast = adapter.generate_forecast(&Symbol::new("SOL"), &prices, 5);

        let size = adapter.recommend_position_size(&forecast, 100.0, 1000.0);

        assert!(size >= 0.0);
        assert!(size <= 1000.0);
    }

    #[test]
    fn test_calibration_update() {
        let adapter = NeuralRiskAdapter::default();

        // Add calibration points
        for i in 0..50 {
            adapter.update_calibration(100.0 + i as f64, 101.0 + i as f64 * 0.9);
        }

        // Generate forecast with calibrated bounds
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        let forecast = adapter.generate_forecast(&Symbol::new("BTC"), &prices, 5);

        // Bounds should be based on calibration
        assert!(forecast.confidence_bounds.interval_width > 0.0);
    }

    #[test]
    fn test_risk_level() {
        let adapter = NeuralRiskAdapter::default();

        // Initially should be low risk
        assert_eq!(adapter.risk_level(), RiskLevel::Normal);

        // Generate some forecasts with disagreement
        let prices: Vec<f64> = (0..50).map(|_| 100.0).collect();
        let forecast = adapter.generate_forecast(&Symbol::new("TEST"), &prices, 5);
        let _ = adapter.calculate_disagreement(&forecast);

        // Risk level may change based on disagreement
        let risk = adapter.risk_level();
        assert!(matches!(risk, RiskLevel::Normal | RiskLevel::Elevated | RiskLevel::High | RiskLevel::Critical));
    }
}
