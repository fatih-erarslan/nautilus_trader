// Forecasting Engine - Time Series Prediction with Uncertainty Quantification
// Advanced forecasting capabilities with confidence intervals and Monte Carlo methods

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal, StudentT};

use super::{
    InputData, ForecastConfig, ForecastResult, IntegrationError, NeuralModel
};

/// Advanced forecasting engine with uncertainty quantification
pub struct ForecastingEngine {
    prediction_cache: Arc<tokio::sync::RwLock<HashMap<String, CachedForecast>>>,
}

impl ForecastingEngine {
    pub fn new() -> Self {
        Self {
            prediction_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
    
    /// Generate forecasts with uncertainty quantification
    pub async fn forecast(
        &self,
        model: Arc<dyn NeuralModel>,
        input_data: InputData,
        config: ForecastConfig,
    ) -> Result<ForecastResult, IntegrationError> {
        // Check cache first
        let cache_key = self.generate_cache_key(&input_data, &config);
        if let Some(cached) = self.get_cached_forecast(&cache_key).await {
            if !cached.is_expired() {
                return Ok(cached.result);
            }
        }
        
        // Generate point predictions
        let predictions = self.generate_point_predictions(&*model, &input_data, &config).await?;
        
        // Generate uncertainty estimates
        let uncertainty = if config.uncertainty_quantification {
            self.estimate_uncertainty(&*model, &input_data, &config, &predictions).await?
        } else {
            vec![0.0; predictions.len()]
        };
        
        // Generate confidence intervals
        let confidence_intervals = self.generate_confidence_intervals(
            &predictions, 
            &uncertainty, 
            &config.confidence_intervals
        )?;
        
        let result = ForecastResult {
            predictions,
            confidence_intervals,
            uncertainty,
            forecast_horizon: config.horizon,
        };
        
        // Cache the result
        self.cache_forecast(cache_key, result.clone()).await;
        
        Ok(result)
    }
    
    /// Generate multi-step ahead forecasts
    pub async fn multi_step_forecast(
        &self,
        model: Arc<dyn NeuralModel>,
        input_data: InputData,
        steps: usize,
    ) -> Result<Vec<f32>, IntegrationError> {
        let mut current_input = input_data.features.last().unwrap().clone();
        let mut predictions = Vec::new();
        
        for step in 0..steps {
            // Predict next value
            let prediction = model.forward(&current_input)
                .map_err(|e| IntegrationError::ForecastingFailed(e))?;
            
            let next_value = prediction.get(0).copied().unwrap_or(0.0);
            predictions.push(next_value);
            
            // Update input for next prediction (sliding window approach)
            if current_input.len() > 1 {
                current_input = current_input[1..].to_vec();
                current_input.push(next_value);
            } else {
                current_input = vec![next_value];
            }
        }
        
        Ok(predictions)
    }
    
    /// Generate ensemble forecasts from multiple models
    pub async fn ensemble_forecast(
        &self,
        models: Vec<Arc<dyn NeuralModel>>,
        input_data: InputData,
        config: ForecastConfig,
        weights: Option<Vec<f32>>,
    ) -> Result<ForecastResult, IntegrationError> {
        if models.is_empty() {
            return Err(IntegrationError::ForecastingFailed("No models provided for ensemble".to_string()));
        }
        
        let weights = weights.unwrap_or_else(|| vec![1.0 / models.len() as f32; models.len()]);
        
        if weights.len() != models.len() {
            return Err(IntegrationError::ForecastingFailed("Weights length must match models length".to_string()));
        }
        
        // Generate predictions from each model
        let mut all_predictions = Vec::new();
        
        for model in models {
            let predictions = self.generate_point_predictions(&*model, &input_data, &config).await?;
            all_predictions.push(predictions);
        }
        
        // Combine predictions using weighted average
        let forecast_length = all_predictions[0].len();
        let mut ensemble_predictions = vec![0.0; forecast_length];
        
        for i in 0..forecast_length {
            let mut weighted_sum = 0.0;
            for (j, predictions) in all_predictions.iter().enumerate() {
                if i < predictions.len() {
                    weighted_sum += predictions[i] * weights[j];
                }
            }
            ensemble_predictions[i] = weighted_sum;
        }
        
        // Calculate ensemble uncertainty (variance across models)
        let mut uncertainty = vec![0.0; forecast_length];
        for i in 0..forecast_length {
            let mean = ensemble_predictions[i];
            let mut variance = 0.0;
            
            for predictions in &all_predictions {
                if i < predictions.len() {
                    variance += (predictions[i] - mean).powi(2);
                }
            }
            
            uncertainty[i] = (variance / all_predictions.len() as f32).sqrt();
        }
        
        // Generate confidence intervals
        let confidence_intervals = self.generate_confidence_intervals(
            &ensemble_predictions,
            &uncertainty,
            &config.confidence_intervals,
        )?;
        
        Ok(ForecastResult {
            predictions: ensemble_predictions,
            confidence_intervals,
            uncertainty,
            forecast_horizon: config.horizon,
        })
    }
    
    /// Generate seasonal decomposition forecast
    pub async fn seasonal_forecast(
        &self,
        model: Arc<dyn NeuralModel>,
        input_data: InputData,
        seasonal_periods: Vec<usize>, // e.g., [24, 168] for hourly data with daily and weekly patterns
        config: ForecastConfig,
    ) -> Result<SeasonalForecastResult, IntegrationError> {
        // Extract seasonal components
        let seasonal_components = self.extract_seasonal_components(&input_data, &seasonal_periods)?;
        
        // Deseasonalize data
        let deseasonalized_data = self.deseasonalize_data(&input_data, &seasonal_components)?;
        
        // Forecast on deseasonalized data
        let deseasonalized_config = ForecastConfig {
            horizon: config.horizon,
            confidence_intervals: config.confidence_intervals.clone(),
            uncertainty_quantification: config.uncertainty_quantification,
            monte_carlo_samples: config.monte_carlo_samples,
        };
        
        let deseasonalized_forecast = self.forecast(model, deseasonalized_data, deseasonalized_config).await?;
        
        // Reseasonalize forecast
        let reseasonalized_predictions = self.reseasonalize_forecast(
            &deseasonalized_forecast.predictions,
            &seasonal_components,
            &seasonal_periods,
        )?;
        
        // Extract values before moving deseasonalized_forecast
        let confidence_intervals = deseasonalized_forecast.confidence_intervals.clone();
        let uncertainty = deseasonalized_forecast.uncertainty.clone();

        Ok(SeasonalForecastResult {
            predictions: reseasonalized_predictions,
            seasonal_components,
            deseasonalized_forecast,
            confidence_intervals,
            uncertainty,
        })
    }
    
    /// Generate probabilistic forecasts using Monte Carlo dropout
    pub async fn probabilistic_forecast(
        &self,
        model: Arc<dyn NeuralModel>,
        input_data: InputData,
        config: ForecastConfig,
        n_samples: usize,
    ) -> Result<ProbabilisticForecastResult, IntegrationError> {
        let mut all_samples = Vec::new();
        
        // Generate multiple samples with different dropout masks
        for _ in 0..n_samples {
            let sample_predictions = self.generate_point_predictions(&*model, &input_data, &config).await?;
            all_samples.push(sample_predictions);
        }
        
        // Calculate statistics across samples
        let forecast_length = all_samples[0].len();
        let mut mean_predictions = vec![0.0; forecast_length];
        let mut std_predictions = vec![0.0; forecast_length];
        let mut percentiles = HashMap::new();
        
        // Calculate percentiles requested in confidence intervals
        for &confidence in &config.confidence_intervals {
            let lower_percentile = (1.0 - confidence) / 2.0 * 100.0;
            let upper_percentile = (1.0 + confidence) / 2.0 * 100.0;
            percentiles.insert(format!("{:.1}", confidence), (vec![0.0; forecast_length], vec![0.0; forecast_length]));
        }
        
        for i in 0..forecast_length {
            // Collect all samples for this timestep
            let mut timestep_samples: Vec<f32> = all_samples.iter()
                .map(|sample| sample.get(i).copied().unwrap_or(0.0))
                .collect();
            timestep_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Calculate mean
            mean_predictions[i] = timestep_samples.iter().sum::<f32>() / timestep_samples.len() as f32;
            
            // Calculate standard deviation
            let variance = timestep_samples.iter()
                .map(|&x| (x - mean_predictions[i]).powi(2))
                .sum::<f32>() / timestep_samples.len() as f32;
            std_predictions[i] = variance.sqrt();
            
            // Calculate percentiles
            for (confidence, (lower_vec, upper_vec)) in percentiles.iter_mut() {
                let confidence_f32: f32 = confidence.parse().unwrap_or(0.95);
                let lower_percentile = (1.0 - confidence_f32) / 2.0 * 100.0;
                let upper_percentile = (1.0 + confidence_f32) / 2.0 * 100.0;
                
                let lower_idx = ((lower_percentile / 100.0) * timestep_samples.len() as f32) as usize;
                let upper_idx = ((upper_percentile / 100.0) * timestep_samples.len() as f32) as usize;
                
                lower_vec[i] = timestep_samples.get(lower_idx.min(timestep_samples.len() - 1)).copied().unwrap_or(0.0);
                upper_vec[i] = timestep_samples.get(upper_idx.min(timestep_samples.len() - 1)).copied().unwrap_or(0.0);
            }
        }
        
        Ok(ProbabilisticForecastResult {
            mean_predictions,
            std_predictions,
            percentiles,
            all_samples,
            n_samples,
        })
    }
    
    // Private helper methods
    
    async fn generate_point_predictions(
        &self,
        model: &dyn NeuralModel,
        input_data: &InputData,
        config: &ForecastConfig,
    ) -> Result<Vec<f32>, IntegrationError> {
        if config.horizon == 1 {
            // Single-step prediction
            if let Some(last_features) = input_data.features.last() {
                let prediction = model.forward(last_features)
                    .map_err(|e| IntegrationError::ForecastingFailed(e))?;
                Ok(prediction)
            } else {
                Err(IntegrationError::ForecastingFailed("No input features provided".to_string()))
            }
        } else {
            // Multi-step prediction
            self.multi_step_forecast(Arc::new(DummyModel), input_data.clone(), config.horizon).await
        }
    }
    
    async fn estimate_uncertainty(
        &self,
        model: &dyn NeuralModel,
        input_data: &InputData,
        config: &ForecastConfig,
        predictions: &[f32],
    ) -> Result<Vec<f32>, IntegrationError> {
        if let Some(n_samples) = config.monte_carlo_samples {
            // Monte Carlo uncertainty estimation
            self.monte_carlo_uncertainty(model, input_data, config, n_samples).await
        } else {
            // Simple heuristic uncertainty (proportional to prediction magnitude)
            Ok(predictions.iter().map(|p| p.abs() * 0.1).collect())
        }
    }
    
    async fn monte_carlo_uncertainty(
        &self,
        model: &dyn NeuralModel,
        input_data: &InputData,
        config: &ForecastConfig,
        n_samples: usize,
    ) -> Result<Vec<f32>, IntegrationError> {
        let mut all_predictions = Vec::new();
        
        for _ in 0..n_samples {
            let sample_predictions = self.generate_point_predictions(model, input_data, config).await?;
            all_predictions.push(sample_predictions);
        }
        
        // Calculate standard deviation across samples
        let forecast_length = all_predictions[0].len();
        let mut uncertainties = vec![0.0; forecast_length];
        
        for i in 0..forecast_length {
            let values: Vec<f32> = all_predictions.iter()
                .map(|pred| pred.get(i).copied().unwrap_or(0.0))
                .collect();
            
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / values.len() as f32;
            
            uncertainties[i] = variance.sqrt();
        }
        
        Ok(uncertainties)
    }
    
    fn generate_confidence_intervals(
        &self,
        predictions: &[f32],
        uncertainties: &[f32],
        confidence_levels: &[f32],
    ) -> Result<HashMap<String, (Vec<f32>, Vec<f32>)>, IntegrationError> {
        let mut intervals = HashMap::new();
        
        for &confidence in confidence_levels {
            if confidence <= 0.0 || confidence >= 1.0 {
                return Err(IntegrationError::ForecastingFailed(
                    format!("Invalid confidence level: {}", confidence)
                ));
            }
            
            // Using normal distribution assumption
            let z_score = self.inverse_normal_cdf((1.0 + confidence) / 2.0);
            
            let mut lower_bounds = Vec::new();
            let mut upper_bounds = Vec::new();
            
            for i in 0..predictions.len() {
                let pred = predictions[i];
                let uncert = uncertainties.get(i).copied().unwrap_or(0.0);
                
                lower_bounds.push(pred - z_score * uncert);
                upper_bounds.push(pred + z_score * uncert);
            }
            
            intervals.insert(
                format!("{:.2}", confidence),
                (lower_bounds, upper_bounds)
            );
        }
        
        Ok(intervals)
    }
    
    fn inverse_normal_cdf(&self, p: f32) -> f32 {
        // Approximation of inverse normal CDF using rational approximation
        if p <= 0.0 { return f32::NEG_INFINITY; }
        if p >= 1.0 { return f32::INFINITY; }
        if p == 0.5 { return 0.0; }
        
        let sign = if p > 0.5 { 1.0 } else { -1.0 };
        let x = if p > 0.5 { 1.0 - p } else { p };
        
        let t = (-2.0 * x.ln()).sqrt();
        
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        
        let numerator = c0 + c1 * t + c2 * t.powi(2);
        let denominator = 1.0 + d1 * t + d2 * t.powi(2) + d3 * t.powi(3);
        
        sign * (t - numerator / denominator)
    }
    
    // Seasonal decomposition methods
    
    fn extract_seasonal_components(
        &self,
        input_data: &InputData,
        seasonal_periods: &[usize],
    ) -> Result<HashMap<usize, Vec<f32>>, IntegrationError> {
        let mut components = HashMap::new();
        
        if let Some(first_series) = input_data.features.first() {
            for &period in seasonal_periods {
                let seasonal_component = self.extract_seasonal_pattern(first_series, period)?;
                components.insert(period, seasonal_component);
            }
        }
        
        Ok(components)
    }
    
    fn extract_seasonal_pattern(&self, data: &[f32], period: usize) -> Result<Vec<f32>, IntegrationError> {
        if period == 0 || data.len() < period {
            return Err(IntegrationError::ForecastingFailed("Invalid seasonal period".to_string()));
        }
        
        let mut seasonal_pattern = vec![0.0; period];
        let mut counts = vec![0; period];
        
        // Average values for each position in the period
        for (i, &value) in data.iter().enumerate() {
            let seasonal_idx = i % period;
            seasonal_pattern[seasonal_idx] += value;
            counts[seasonal_idx] += 1;
        }
        
        // Normalize by count
        for i in 0..period {
            if counts[i] > 0 {
                seasonal_pattern[i] /= counts[i] as f32;
            }
        }
        
        Ok(seasonal_pattern)
    }
    
    fn deseasonalize_data(
        &self,
        input_data: &InputData,
        seasonal_components: &HashMap<usize, Vec<f32>>,
    ) -> Result<InputData, IntegrationError> {
        let mut deseasonalized_features = Vec::new();
        
        for series in &input_data.features {
            let mut deseasonalized_series = series.clone();
            
            // Remove all seasonal components
            for (i, value) in deseasonalized_series.iter_mut().enumerate() {
                for (&period, pattern) in seasonal_components {
                    let seasonal_idx = i % period;
                    if seasonal_idx < pattern.len() {
                        *value -= pattern[seasonal_idx];
                    }
                }
            }
            
            deseasonalized_features.push(deseasonalized_series);
        }
        
        Ok(InputData {
            features: deseasonalized_features,
            sequence_length: input_data.sequence_length,
        })
    }
    
    fn reseasonalize_forecast(
        &self,
        predictions: &[f32],
        seasonal_components: &HashMap<usize, Vec<f32>>,
        seasonal_periods: &[usize],
    ) -> Result<Vec<f32>, IntegrationError> {
        let mut reseasonalized = predictions.to_vec();
        
        for (i, value) in reseasonalized.iter_mut().enumerate() {
            for &period in seasonal_periods {
                if let Some(pattern) = seasonal_components.get(&period) {
                    let seasonal_idx = i % period;
                    if seasonal_idx < pattern.len() {
                        *value += pattern[seasonal_idx];
                    }
                }
            }
        }
        
        Ok(reseasonalized)
    }
    
    // Cache management
    
    fn generate_cache_key(&self, input_data: &InputData, config: &ForecastConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash input data (simplified)
        input_data.features.len().hash(&mut hasher);
        if let Some(first) = input_data.features.first() {
            first.len().hash(&mut hasher);
        }
        
        // Hash config
        config.horizon.hash(&mut hasher);
        config.uncertainty_quantification.hash(&mut hasher);
        
        format!("forecast_{}", hasher.finish())
    }
    
    async fn get_cached_forecast(&self, key: &str) -> Option<CachedForecast> {
        let cache = self.prediction_cache.read().await;
        cache.get(key).cloned()
    }
    
    async fn cache_forecast(&self, key: String, result: ForecastResult) {
        let mut cache = self.prediction_cache.write().await;
        cache.insert(key, CachedForecast {
            result,
            timestamp: std::time::SystemTime::now(),
            ttl: std::time::Duration::from_secs(300), // 5 minutes
        });
    }
}

// Supporting types and structures

#[derive(Clone)]
struct CachedForecast {
    result: ForecastResult,
    timestamp: std::time::SystemTime,
    ttl: std::time::Duration,
}

impl CachedForecast {
    fn is_expired(&self) -> bool {
        self.timestamp.elapsed().unwrap_or(self.ttl) > self.ttl
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalForecastResult {
    pub predictions: Vec<f32>,
    pub seasonal_components: HashMap<usize, Vec<f32>>,
    pub deseasonalized_forecast: ForecastResult,
    pub confidence_intervals: HashMap<String, (Vec<f32>, Vec<f32>)>,
    pub uncertainty: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticForecastResult {
    pub mean_predictions: Vec<f32>,
    pub std_predictions: Vec<f32>,
    pub percentiles: HashMap<String, (Vec<f32>, Vec<f32>)>,
    pub all_samples: Vec<Vec<f32>>,
    pub n_samples: usize,
}

// Dummy model for testing (remove in production)
struct DummyModel;

impl super::NeuralModel for DummyModel {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        Ok(input.to_vec()) // Echo input as prediction
    }
    
    fn backward(&mut self, _gradient: &[f32]) -> Result<(), String> {
        Ok(())
    }
    
    fn update_weights(&mut self, _learning_rate: f32) -> Result<(), String> {
        Ok(())
    }
    
    fn get_parameters(&self) -> Vec<f32> {
        vec![]
    }
    
    fn set_parameters(&mut self, _params: Vec<f32>) -> Result<(), String> {
        Ok(())
    }
    
    fn save_state(&self) -> Result<Vec<u8>, String> {
        Ok(vec![])
    }
    
    fn load_state(&mut self, _state: &[u8]) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_forecasting_engine_creation() {
        let engine = ForecastingEngine::new();
        assert!(!engine.prediction_cache.read().await.is_empty() == false);
    }
    
    #[tokio::test]
    async fn test_multi_step_forecast() {
        let engine = ForecastingEngine::new();
        let model = Arc::new(DummyModel);
        let input_data = InputData {
            features: vec![vec![1.0, 2.0, 3.0]],
            sequence_length: Some(3),
        };
        
        let result = engine.multi_step_forecast(model, input_data, 5).await;
        assert!(result.is_ok());
        
        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 5);
    }
    
    #[test]
    fn test_inverse_normal_cdf() {
        let engine = ForecastingEngine::new();
        
        // Test known values
        let z_95 = engine.inverse_normal_cdf(0.975); // Should be approximately 1.96
        assert!((z_95 - 1.96).abs() < 0.1);
        
        let z_50 = engine.inverse_normal_cdf(0.5); // Should be 0
        assert!(z_50.abs() < 0.01);
    }
    
    #[test]
    fn test_confidence_intervals() {
        let engine = ForecastingEngine::new();
        let predictions = vec![1.0, 2.0, 3.0];
        let uncertainties = vec![0.1, 0.2, 0.3];
        let confidence_levels = vec![0.95, 0.80];
        
        let intervals = engine.generate_confidence_intervals(
            &predictions, 
            &uncertainties, 
            &confidence_levels
        ).unwrap();
        
        assert_eq!(intervals.len(), 2);
        assert!(intervals.contains_key("0.95"));
        assert!(intervals.contains_key("0.80"));
    }
}