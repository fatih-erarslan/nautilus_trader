//! Neural forecasting engine for time series prediction.

use crate::adapter::NeuralFeatures;
use crate::config::{NeuralBridgeConfig, NeuralModelType};
use crate::error::{NeuralBridgeError, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

/// Result of a neural forecast
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point predictions for each horizon step
    pub predictions: Vec<f64>,
    /// Lower confidence bound
    pub lower_bound: Vec<f64>,
    /// Upper confidence bound
    pub upper_bound: Vec<f64>,
    /// Prediction variance/uncertainty
    pub variance: Vec<f64>,
    /// Model that produced this forecast
    pub model_type: NeuralModelType,
    /// Confidence level used (e.g., 0.95)
    pub confidence_level: f64,
    /// Feature importance scores (if available)
    pub feature_importance: Option<HashMap<String, f64>>,
    /// Forecast horizon (number of steps ahead)
    pub horizon: usize,
    /// Timestamp of forecast generation
    pub generated_at: f64,
}

impl ForecastResult {
    /// Get the primary prediction (first horizon step)
    pub fn primary_prediction(&self) -> f64 {
        self.predictions.first().copied().unwrap_or(0.0)
    }

    /// Get the confidence interval width at a specific horizon
    pub fn interval_width(&self, horizon_idx: usize) -> f64 {
        if horizon_idx < self.upper_bound.len() && horizon_idx < self.lower_bound.len() {
            self.upper_bound[horizon_idx] - self.lower_bound[horizon_idx]
        } else {
            0.0
        }
    }

    /// Calculate prediction quality score based on uncertainty
    pub fn quality_score(&self) -> f64 {
        if self.variance.is_empty() {
            return 0.5;
        }
        let avg_variance = self.variance.iter().sum::<f64>() / self.variance.len() as f64;
        // Lower variance = higher quality (inverse relationship)
        1.0 / (1.0 + avg_variance)
    }
}

/// Neural forecasting engine
pub struct NeuralForecastEngine {
    /// Configuration
    config: NeuralBridgeConfig,
    /// Model weights cache (for future GPU model loading)
    model_cache: Arc<RwLock<HashMap<NeuralModelType, ModelState>>>,
    /// Historical forecast accuracy tracking
    accuracy_tracker: Arc<RwLock<AccuracyTracker>>,
}

/// State of a loaded model
#[derive(Debug, Clone)]
struct ModelState {
    /// Whether model is loaded and ready
    ready: bool,
    /// Last inference latency in microseconds
    last_latency_us: u64,
    /// Total predictions made
    prediction_count: u64,
}

/// Tracks forecast accuracy for model selection
#[derive(Debug, Clone, Default)]
struct AccuracyTracker {
    /// Mean Absolute Error per model
    mae_by_model: HashMap<NeuralModelType, f64>,
    /// Root Mean Square Error per model
    rmse_by_model: HashMap<NeuralModelType, f64>,
    /// Sample count per model
    samples_by_model: HashMap<NeuralModelType, usize>,
}

impl NeuralForecastEngine {
    /// Create a new forecasting engine
    pub fn new(config: NeuralBridgeConfig) -> Self {
        let mut model_cache = HashMap::new();

        // Initialize model states for configured models
        for model_type in &config.ensemble_models {
            model_cache.insert(
                *model_type,
                ModelState {
                    ready: true, // In standalone mode, models are always "ready"
                    last_latency_us: 0,
                    prediction_count: 0,
                },
            );
        }

        Self {
            config,
            model_cache: Arc::new(RwLock::new(model_cache)),
            accuracy_tracker: Arc::new(RwLock::new(AccuracyTracker::default())),
        }
    }

    /// Generate forecast from neural features
    pub async fn forecast(&self, features: &NeuralFeatures) -> Result<ForecastResult> {
        let horizon = self.config.forecast_horizon;

        // Validate input
        if features.feature_matrix.nrows() < self.config.min_sequence_length {
            return Err(NeuralBridgeError::InsufficientData {
                required: self.config.min_sequence_length,
                actual: features.feature_matrix.nrows(),
            });
        }

        debug!(
            horizon = horizon,
            input_length = features.feature_matrix.nrows(),
            "Generating neural forecast"
        );

        // Use primary model from ensemble
        let model_type = self.config.ensemble_models.first()
            .copied()
            .unwrap_or(NeuralModelType::NHITS);

        // Generate predictions using the selected model
        let predictions = self.run_model_inference(model_type, features, horizon).await?;

        // Calculate confidence intervals
        let (lower, upper, variance) = self.calculate_confidence_intervals(
            &predictions,
            features,
            self.config.confidence_level,
        );

        // Update model stats
        {
            let mut cache = self.model_cache.write().await;
            if let Some(state) = cache.get_mut(&model_type) {
                state.prediction_count += 1;
            }
        }

        Ok(ForecastResult {
            predictions,
            lower_bound: lower,
            upper_bound: upper,
            variance,
            model_type,
            confidence_level: self.config.confidence_level,
            feature_importance: None,
            horizon,
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        })
    }

    /// Run inference with a specific model type
    async fn run_model_inference(
        &self,
        model_type: NeuralModelType,
        features: &NeuralFeatures,
        horizon: usize,
    ) -> Result<Vec<f64>> {
        trace!(model = %model_type, "Running model inference");

        // In standalone mode, use statistical forecasting methods
        // These serve as baselines until full Neural Trader integration
        match model_type {
            NeuralModelType::NHITS => self.nhits_inference(features, horizon),
            NeuralModelType::LSTMAttention => self.lstm_inference(features, horizon),
            NeuralModelType::Transformer => self.transformer_inference(features, horizon),
            NeuralModelType::GRU => self.gru_inference(features, horizon),
            NeuralModelType::TCN => self.tcn_inference(features, horizon),
            NeuralModelType::DeepAR => self.deepar_inference(features, horizon),
            NeuralModelType::NBeats => self.nbeats_inference(features, horizon),
            NeuralModelType::Prophet => self.prophet_inference(features, horizon),
        }
    }

    /// NHITS-style hierarchical interpolation forecast
    fn nhits_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        let prices = &features.prices;
        if prices.len() < 2 {
            return Err(NeuralBridgeError::InsufficientData {
                required: 2,
                actual: prices.len(),
            });
        }

        // Multi-scale trend decomposition (NHITS-inspired)
        let mut predictions = Vec::with_capacity(horizon);

        // Calculate trends at different scales
        let short_trend = self.calculate_trend(&prices[prices.len().saturating_sub(12)..]);
        let medium_trend = self.calculate_trend(&prices[prices.len().saturating_sub(48)..]);
        let long_trend = self.calculate_trend(prices);

        // Weighted combination (hierarchical interpolation)
        let last_price = *prices.last().unwrap();
        for h in 1..=horizon {
            let h_f64 = h as f64;
            // Blend trends based on horizon
            let weight_short = (-0.1 * h_f64).exp();
            let weight_medium = 1.0 - weight_short;

            let blended_trend = weight_short * short_trend
                + weight_medium * 0.6 * medium_trend
                + weight_medium * 0.4 * long_trend;

            predictions.push(last_price * (1.0 + blended_trend * h_f64));
        }

        Ok(predictions)
    }

    /// LSTM-style sequential forecast with attention weighting
    fn lstm_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        let prices = &features.prices;
        let returns = &features.returns;

        if prices.len() < 2 {
            return Err(NeuralBridgeError::InsufficientData {
                required: 2,
                actual: prices.len(),
            });
        }

        // Attention-weighted historical average
        let attention_weights = self.compute_attention_weights(prices.len());

        let mut predictions = Vec::with_capacity(horizon);
        let last_price = *prices.last().unwrap();

        // Compute weighted average return
        let avg_return = if !returns.is_empty() {
            returns.iter()
                .rev()
                .take(attention_weights.len())
                .zip(attention_weights.iter())
                .map(|(r, w)| r * w)
                .sum::<f64>()
        } else {
            0.0
        };

        // Project forward with decay
        for h in 1..=horizon {
            let decay = (-0.05 * h as f64).exp();
            let projected_return = avg_return * decay;
            let prev_price = if h == 1 {
                last_price
            } else {
                predictions[h - 2]
            };
            predictions.push(prev_price * (1.0 + projected_return));
        }

        Ok(predictions)
    }

    /// Transformer-style forecast using self-attention patterns
    fn transformer_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        let prices = &features.prices;

        if prices.len() < 4 {
            return Err(NeuralBridgeError::InsufficientData {
                required: 4,
                actual: prices.len(),
            });
        }

        // Multi-head attention simulation (pattern matching)
        let patterns = self.extract_price_patterns(prices);
        let last_price = *prices.last().unwrap();

        let mut predictions = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            // Combine patterns with positional weighting
            let pattern_pred = patterns.iter()
                .enumerate()
                .map(|(i, p)| p * (1.0 / (1 + i) as f64))
                .sum::<f64>()
                / patterns.len() as f64;

            let base = if h == 1 { last_price } else { predictions[h - 2] };
            predictions.push(base * (1.0 + pattern_pred * 0.01));
        }

        Ok(predictions)
    }

    /// GRU-style gated forecast
    fn gru_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        // Similar to LSTM but with simplified gating
        self.lstm_inference(features, horizon)
    }

    /// TCN-style dilated causal convolution forecast
    fn tcn_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        let prices = &features.prices;

        if prices.len() < 8 {
            return Err(NeuralBridgeError::InsufficientData {
                required: 8,
                actual: prices.len(),
            });
        }

        // Dilated convolution effect (multi-scale feature extraction)
        let scale1 = self.moving_average(prices, 2);
        let scale2 = self.moving_average(prices, 4);
        let scale4 = self.moving_average(prices, 8);

        let last_price = *prices.last().unwrap();
        let trend = (scale1 - scale4) / scale4.max(1e-8);

        let mut predictions = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let base = if h == 1 { last_price } else { predictions[h - 2] };
            let decay = (-0.1 * h as f64).exp();
            predictions.push(base * (1.0 + trend * decay));
        }

        Ok(predictions)
    }

    /// DeepAR-style probabilistic forecast
    fn deepar_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        // DeepAR focuses on probabilistic forecasts
        // For point predictions, use LSTM backbone
        self.lstm_inference(features, horizon)
    }

    /// N-BEATS-style pure MLP forecast with decomposition
    fn nbeats_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        let prices = &features.prices;

        if prices.len() < 2 {
            return Err(NeuralBridgeError::InsufficientData {
                required: 2,
                actual: prices.len(),
            });
        }

        // Trend + Seasonality decomposition (N-BEATS style)
        let trend = self.calculate_trend(prices);
        let seasonality = self.estimate_seasonality(prices);

        let last_price = *prices.last().unwrap();
        let mut predictions = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let trend_component = last_price * (1.0 + trend * h as f64);
            let seasonal_component = seasonality * (2.0 * std::f64::consts::PI * h as f64 / 24.0).sin();
            predictions.push(trend_component + seasonal_component);
        }

        Ok(predictions)
    }

    /// Prophet-style forecast with trend changepoints
    fn prophet_inference(&self, features: &NeuralFeatures, horizon: usize) -> Result<Vec<f64>> {
        // Simplified Prophet: piecewise linear trend + seasonality
        self.nbeats_inference(features, horizon)
    }

    /// Calculate trend from price series
    fn calculate_trend(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        // Linear regression slope
        let n = prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = prices.iter().sum::<f64>() / n;

        let numerator: f64 = prices.iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = (0..prices.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        let slope = numerator / denominator;
        // Normalize to return space
        slope / y_mean.max(1e-8)
    }

    /// Compute attention weights (softmax over positions)
    fn compute_attention_weights(&self, seq_len: usize) -> Vec<f64> {
        let weights: Vec<f64> = (0..seq_len.min(24))
            .map(|i| (-(i as f64) * 0.2).exp())
            .collect();

        let sum: f64 = weights.iter().sum();
        weights.into_iter().map(|w| w / sum.max(1e-8)).collect()
    }

    /// Extract repeating price patterns
    fn extract_price_patterns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 4 {
            return vec![0.0];
        }

        // Calculate returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(1e-8))
            .collect();

        // Find patterns (simplified autocorrelation)
        let mut patterns = Vec::new();
        for lag in 1..=4.min(returns.len() / 2) {
            let correlation = self.autocorrelation(&returns, lag);
            patterns.push(correlation);
        }

        patterns
    }

    /// Calculate autocorrelation at given lag
    fn autocorrelation(&self, series: &[f64], lag: usize) -> f64 {
        if series.len() <= lag {
            return 0.0;
        }

        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let var: f64 = series.iter().map(|x| (x - mean).powi(2)).sum();

        if var.abs() < 1e-10 {
            return 0.0;
        }

        let cov: f64 = series[..series.len() - lag]
            .iter()
            .zip(&series[lag..])
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum();

        cov / var
    }

    /// Calculate moving average
    fn moving_average(&self, prices: &[f64], window: usize) -> f64 {
        let start = prices.len().saturating_sub(window);
        let slice = &prices[start..];
        slice.iter().sum::<f64>() / slice.len().max(1) as f64
    }

    /// Estimate seasonality component
    fn estimate_seasonality(&self, prices: &[f64]) -> f64 {
        if prices.len() < 24 {
            return 0.0;
        }

        // Detrend and find periodic component
        let trend = self.calculate_trend(prices);
        let detrended: Vec<f64> = prices.iter()
            .enumerate()
            .map(|(i, &p)| p - prices[0] * (1.0 + trend * i as f64))
            .collect();

        // Amplitude of oscillation
        let max = detrended.iter().cloned().fold(f64::MIN, f64::max);
        let min = detrended.iter().cloned().fold(f64::MAX, f64::min);

        (max - min) / 2.0
    }

    /// Calculate confidence intervals using conformal prediction approach
    fn calculate_confidence_intervals(
        &self,
        predictions: &[f64],
        features: &NeuralFeatures,
        confidence_level: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = predictions.len();

        // Estimate prediction variance from historical volatility
        let base_variance = if !features.volatility.is_empty() {
            features.volatility.iter().sum::<f64>() / features.volatility.len() as f64
        } else {
            0.02 // Default 2% volatility
        };

        // Z-score for confidence level
        let z_score = self.normal_quantile((1.0 + confidence_level) / 2.0);

        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);
        let mut variance = Vec::with_capacity(n);

        for (h, pred) in predictions.iter().enumerate() {
            // Variance grows with horizon (uncertainty increases)
            let horizon_factor = 1.0 + 0.1 * h as f64;
            let var_h = base_variance.powi(2) * horizon_factor;
            let std_h = var_h.sqrt();

            // Scale interval by prediction magnitude, with minimum bound to ensure
            // lower < prediction < upper even for small predictions
            let pred_scale = pred.abs().max(1.0); // Minimum scale of 1.0
            let interval = (z_score * std_h * pred_scale).max(0.001);

            lower.push(pred - interval);
            upper.push(pred + interval);
            variance.push(var_h);
        }

        (lower, upper, variance)
    }

    /// Approximate inverse normal CDF (quantile function)
    fn normal_quantile(&self, p: f64) -> f64 {
        // Rational approximation for normal quantile
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        let a = [
            -3.969_683_028_665_376e1,
            2.209_460_984_245_205e2,
            -2.759_285_104_469_687e2,
            1.383_577_518_672_690e2,
            -3.066_479_806_614_716e1,
            2.506_628_277_459_239e0,
        ];
        let b = [
            -5.447_609_879_822_406e1,
            1.615_858_368_580_409e2,
            -1.556_989_798_598_866e2,
            6.680_131_188_771_972e1,
            -1.328_068_155_288_572e1,
        ];

        let q = p - 0.5;
        if q.abs() <= 0.425 {
            let r = 0.180625 - q * q;
            q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let r = if q < 0.0 { p } else { 1.0 - p };
            let r = (-r.ln()).sqrt();

            let c = [
                -7.784_894_002_430_293e-3,
                -3.223_964_580_411_365e-1,
                -2.400_758_277_161_838e0,
                -2.549_732_539_343_734e0,
                4.374_664_141_464_968e0,
                2.938_163_982_698_783e0,
            ];
            let d = [
                7.784_695_709_041_462e-3,
                3.224_671_290_700_398e-1,
                2.445_134_137_142_996e0,
                3.754_408_661_907_416e0,
            ];

            let result = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
                / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0);

            if q < 0.0 { -result } else { result }
        }
    }

    /// Update accuracy tracker with actual vs predicted values
    pub async fn update_accuracy(
        &self,
        model_type: NeuralModelType,
        predicted: f64,
        actual: f64,
    ) {
        let error = (predicted - actual).abs();
        let squared_error = error.powi(2);

        let mut tracker = self.accuracy_tracker.write().await;

        // Update MAE
        let count = tracker.samples_by_model.entry(model_type).or_insert(0);
        *count += 1;
        let n = *count as f64;

        let mae = tracker.mae_by_model.entry(model_type).or_insert(0.0);
        *mae = *mae + (error - *mae) / n;

        let rmse = tracker.rmse_by_model.entry(model_type).or_insert(0.0);
        *rmse = (*rmse * (n - 1.0) / n + squared_error / n).sqrt();
    }

    /// Get model accuracy statistics
    pub async fn get_accuracy_stats(&self) -> HashMap<NeuralModelType, (f64, f64)> {
        let tracker = self.accuracy_tracker.read().await;
        let mut stats = HashMap::new();

        for model_type in &self.config.ensemble_models {
            let mae = tracker.mae_by_model.get(model_type).copied().unwrap_or(f64::NAN);
            let rmse = tracker.rmse_by_model.get(model_type).copied().unwrap_or(f64::NAN);
            stats.insert(*model_type, (mae, rmse));
        }

        stats
    }
}

impl Clone for NeuralForecastEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            model_cache: Arc::clone(&self.model_cache),
            accuracy_tracker: Arc::clone(&self.accuracy_tracker),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{MarketFeed, NeuralDataAdapter};
    use crate::config::NeuralBridgeConfig;
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
    async fn test_forecast_engine_creation() {
        let config = NeuralBridgeConfig::default();
        let engine = NeuralForecastEngine::new(config);
        assert!(!engine.config.ensemble_models.is_empty());
    }

    #[tokio::test]
    async fn test_nhits_forecast() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 10,
            forecast_horizon: 6,
            ..Default::default()
        };
        let engine = NeuralForecastEngine::new(config);

        // Generate test price series (upward trend)
        let prices: Vec<f64> = (0..24).map(|i| 100.0 + i as f64 * 0.5).collect();
        let features = make_test_features(prices);

        let result = engine.forecast(&features).await.unwrap();

        assert_eq!(result.predictions.len(), 6);
        assert_eq!(result.lower_bound.len(), 6);
        assert_eq!(result.upper_bound.len(), 6);
        // Predictions should continue upward trend
        assert!(result.predictions[0] > 100.0);
    }

    #[tokio::test]
    async fn test_confidence_intervals() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 5,
            forecast_horizon: 4,
            confidence_level: 0.95,
            ..Default::default()
        };
        let engine = NeuralForecastEngine::new(config);

        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let features = make_test_features(prices);

        let result = engine.forecast(&features).await.unwrap();

        // Lower bound should be less than prediction
        for (i, pred) in result.predictions.iter().enumerate() {
            assert!(result.lower_bound[i] < *pred);
            assert!(result.upper_bound[i] > *pred);
        }
    }

    #[tokio::test]
    async fn test_insufficient_data_error() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 24,
            ..Default::default()
        };
        let engine = NeuralForecastEngine::new(config);

        let features = make_test_features(vec![100.0, 101.0, 102.0]);

        let result = engine.forecast(&features).await;
        assert!(matches!(
            result,
            Err(NeuralBridgeError::InsufficientData { .. })
        ));
    }

    #[tokio::test]
    async fn test_accuracy_tracking() {
        let config = NeuralBridgeConfig::default();
        let engine = NeuralForecastEngine::new(config);

        engine.update_accuracy(NeuralModelType::NHITS, 100.0, 101.0).await;
        engine.update_accuracy(NeuralModelType::NHITS, 102.0, 100.0).await;

        let stats = engine.get_accuracy_stats().await;
        let (mae, _rmse) = stats.get(&NeuralModelType::NHITS).unwrap();

        assert!(*mae > 0.0);
    }

    #[test]
    fn test_forecast_result_methods() {
        let result = ForecastResult {
            predictions: vec![100.0, 101.0, 102.0],
            lower_bound: vec![98.0, 98.5, 99.0],
            upper_bound: vec![102.0, 103.5, 105.0],
            variance: vec![0.01, 0.02, 0.03],
            model_type: NeuralModelType::NHITS,
            confidence_level: 0.95,
            feature_importance: None,
            horizon: 3,
            generated_at: 0.0,
        };

        assert_eq!(result.primary_prediction(), 100.0);
        assert_eq!(result.interval_width(0), 4.0);
        assert!(result.quality_score() > 0.0);
    }
}
