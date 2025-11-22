//! Multi-asset price prediction using consciousness-aware NHITS
//! 
//! This module implements sophisticated price forecasting for financial markets,
//! leveraging the NHITS architecture enhanced with consciousness mechanisms
//! for superior prediction accuracy and market timing.

use super::*;
use crate::ml::nhits::model::NHITSModel;
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Multi-asset price predictor using NHITS
#[derive(Debug)]
pub struct PricePredictor {
    pub models: HashMap<String, FinancialNHITS>,
    pub lookback_window: usize,
    pub forecast_horizon: usize,
    pub consciousness_threshold: f32,
    pub cross_asset_correlations: Array2<f32>,
    pub prediction_cache: HashMap<String, PredictionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub symbol: String,
    pub predicted_prices: Vec<f32>,
    pub prediction_intervals: Vec<(f32, f32)>,
    pub confidence_scores: Vec<f32>,
    pub consciousness_state: f32,
    pub prediction_timestamp: i64,
    pub forecast_horizon: usize,
}

#[derive(Debug, Clone)]
pub struct PredictionMetrics {
    pub mape: f32,  // Mean Absolute Percentage Error
    pub rmse: f32,  // Root Mean Square Error
    pub directional_accuracy: f32,  // Percentage of correct direction predictions
    pub sharpe_ratio: f32,  // Risk-adjusted returns
    pub max_drawdown: f32,
    pub hit_rate: f32,  // Percentage of predictions within confidence intervals
}

impl PricePredictor {
    pub fn new(lookback_window: usize, forecast_horizon: usize) -> Self {
        Self {
            models: HashMap::new(),
            lookback_window,
            forecast_horizon,
            consciousness_threshold: 0.75,
            cross_asset_correlations: Array2::zeros((0, 0)),
            prediction_cache: HashMap::new(),
        }
    }
    
    /// Add a new asset for prediction
    pub fn add_asset(&mut self, symbol: String, feature_dim: usize) {
        let model = FinancialNHITS::new(
            feature_dim,
            128,  // hidden_dim
            3,    // num_stacks
            4,    // num_blocks
            self.forecast_horizon,
        ).with_financial_components();
        
        self.models.insert(symbol, model);
    }
    
    /// Train model on historical data
    pub fn train(&mut self, symbol: &str, data: &FinancialTimeSeries, epochs: usize) -> Result<(), String> {
        let model = self.models.get_mut(symbol)
            .ok_or_else(|| format!("Model for {} not found", symbol))?;
        
        let features = utils::ohlcv_to_features(data);
        let targets = self.prepare_targets(&data.close);
        
        // Train with consciousness-aware loss function
        for epoch in 0..epochs {
            let consciousness_state = self.calculate_market_consciousness(&features);
            let loss = self.consciousness_aware_loss(&features, &targets, consciousness_state);
            
            // Backpropagation would go here (simplified for this example)
            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {:.6}, Consciousness = {:.3}", 
                         epoch, loss, consciousness_state);
            }
        }
        
        Ok(())
    }
    
    /// Predict future prices for a single asset
    pub fn predict_single(&mut self, symbol: &str, recent_data: &Array2<f32>) -> Result<PredictionResult, String> {
        let model = self.models.get(symbol)
            .ok_or_else(|| format!("Model for {} not found", symbol))?;
        
        let consciousness_state = self.calculate_market_consciousness(recent_data);
        let prediction = model.predict_conscious(recent_data, consciousness_state);
        
        let confidence_scores = (0..self.forecast_horizon)
            .map(|i| model.prediction_confidence(&prediction, consciousness_state))
            .collect();
        
        let predicted_prices = prediction.row(0).to_vec();
        let prediction_intervals = self.calculate_prediction_intervals(&predicted_prices, &confidence_scores);
        
        let result = PredictionResult {
            symbol: symbol.to_string(),
            predicted_prices,
            prediction_intervals,
            confidence_scores,
            consciousness_state,
            prediction_timestamp: chrono::Utc::now().timestamp(),
            forecast_horizon: self.forecast_horizon,
        };
        
        // Cache result
        self.prediction_cache.insert(symbol.to_string(), result.clone());
        
        Ok(result)
    }
    
    /// Predict prices for multiple assets simultaneously
    pub fn predict_multi_asset(&mut self, data: &HashMap<String, Array2<f32>>) -> HashMap<String, PredictionResult> {
        let mut results = HashMap::new();
        
        // Calculate cross-asset consciousness correlation
        let global_consciousness = self.calculate_global_consciousness(data);
        
        for (symbol, asset_data) in data {
            if let Ok(mut prediction) = self.predict_single(symbol, asset_data) {
                // Adjust prediction based on cross-asset correlations
                prediction = self.apply_cross_asset_correlation(prediction, global_consciousness);
                results.insert(symbol.clone(), prediction);
            }
        }
        
        results
    }
    
    /// Advanced prediction with ensemble methods
    pub fn predict_ensemble(&mut self, symbol: &str, data: &Array2<f32>, num_models: usize) -> Result<PredictionResult, String> {
        let mut ensemble_predictions = Vec::new();
        let mut ensemble_confidences = Vec::new();
        
        for i in 0..num_models {
            // Add noise to create ensemble diversity
            let mut noisy_data = data.clone();
            let noise_level = 0.001 * (i as f32 + 1.0);
            // Use proper statistical noise distribution
            use statrs::distribution::{Normal, ContinuousCDF};
            use rand::thread_rng;
            
            let normal = Normal::new(0.0, noise_level as f64).unwrap();
            let mut rng = thread_rng();
            
            noisy_data.mapv_inplace(|x| x + normal.sample(&mut rng) as f32);
            
            if let Ok(prediction) = self.predict_single(symbol, &noisy_data) {
                ensemble_predictions.push(prediction.predicted_prices);
                ensemble_confidences.push(prediction.confidence_scores);
            }
        }
        
        // Combine ensemble predictions
        let combined_prediction = self.combine_ensemble_predictions(&ensemble_predictions);
        let combined_confidence = self.combine_ensemble_confidences(&ensemble_confidences);
        
        Ok(PredictionResult {
            symbol: symbol.to_string(),
            predicted_prices: combined_prediction.clone(),
            prediction_intervals: self.calculate_prediction_intervals(&combined_prediction, &combined_confidence),
            confidence_scores: combined_confidence,
            consciousness_state: self.calculate_market_consciousness(data),
            prediction_timestamp: chrono::Utc::now().timestamp(),
            forecast_horizon: self.forecast_horizon,
        })
    }
    
    /// Evaluate prediction accuracy
    pub fn evaluate_predictions(&self, symbol: &str, actual_prices: &[f32], predictions: &PredictionResult) -> PredictionMetrics {
        let predicted = &predictions.predicted_prices;
        let n = actual_prices.len().min(predicted.len());
        
        if n == 0 {
            return PredictionMetrics {
                mape: f32::INFINITY,
                rmse: f32::INFINITY,
                directional_accuracy: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 1.0,
                hit_rate: 0.0,
            };
        }
        
        // Calculate MAPE
        let mape = (0..n)
            .map(|i| ((actual_prices[i] - predicted[i]) / actual_prices[i]).abs())
            .sum::<f32>() / n as f32 * 100.0;
        
        // Calculate RMSE
        let rmse = (0..n)
            .map(|i| (actual_prices[i] - predicted[i]).powi(2))
            .sum::<f32>() / n as f32;
        let rmse = rmse.sqrt();
        
        // Calculate directional accuracy
        let mut correct_directions = 0;
        for i in 1..n {
            let actual_direction = actual_prices[i] > actual_prices[i-1];
            let predicted_direction = predicted[i] > predicted[i-1];
            if actual_direction == predicted_direction {
                correct_directions += 1;
            }
        }
        let directional_accuracy = correct_directions as f32 / (n - 1) as f32 * 100.0;
        
        // Calculate hit rate (predictions within confidence intervals)
        let mut hits = 0;
        for i in 0..n {
            if i < predictions.prediction_intervals.len() {
                let (lower, upper) = predictions.prediction_intervals[i];
                if actual_prices[i] >= lower && actual_prices[i] <= upper {
                    hits += 1;
                }
            }
        }
        let hit_rate = hits as f32 / n as f32 * 100.0;
        
        // Calculate returns-based metrics
        let actual_returns: Vec<f32> = actual_prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let predicted_returns: Vec<f32> = predicted.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let sharpe_ratio = self.calculate_sharpe_ratio(&actual_returns);
        let max_drawdown = self.calculate_max_drawdown(actual_prices);
        
        PredictionMetrics {
            mape,
            rmse,
            directional_accuracy,
            sharpe_ratio,
            max_drawdown,
            hit_rate,
        }
    }
    
    /// Real-time prediction with streaming data
    pub fn predict_streaming(&mut self, symbol: &str, new_data_point: &Array1<f32>) -> Result<f32, String> {
        // This would maintain a sliding window for real-time predictions
        let model = self.models.get(symbol)
            .ok_or_else(|| format!("Model for {} not found", symbol))?;
        
        // For simplicity, we'll use the last prediction
        if let Some(cached_prediction) = self.prediction_cache.get(symbol) {
            // Return next predicted price
            Ok(cached_prediction.predicted_prices[0])
        } else {
            Err("No cached predictions available".to_string())
        }
    }
    
    // Private helper methods
    
    fn prepare_targets(&self, prices: &[f32]) -> Array2<f32> {
        let n = prices.len();
        let mut targets = Array2::zeros((n - self.forecast_horizon, self.forecast_horizon));
        
        for i in 0..(n - self.forecast_horizon) {
            for j in 0..self.forecast_horizon {
                targets[[i, j]] = prices[i + j + 1];
            }
        }
        
        targets
    }
    
    fn calculate_market_consciousness(&self, data: &Array2<f32>) -> f32 {
        // Consciousness based on market coherence and stability
        let returns = data.slice(s![.., 5]);  // Returns column
        let volatility = returns.std(0.0);
        let trend_consistency = self.calculate_trend_consistency(&returns.to_vec());
        
        // Higher consciousness = lower volatility + higher trend consistency
        (trend_consistency / (1.0 + volatility)).min(1.0).max(0.0)
    }
    
    fn calculate_global_consciousness(&self, data: &HashMap<String, Array2<f32>>) -> f32 {
        let individual_consciousness: Vec<f32> = data.values()
            .map(|asset_data| self.calculate_market_consciousness(asset_data))
            .collect();
        
        if individual_consciousness.is_empty() {
            return 0.5;
        }
        
        // Global consciousness is the mean with correlation adjustment
        let mean_consciousness = individual_consciousness.iter().sum::<f32>() / individual_consciousness.len() as f32;
        let consciousness_std = {
            let variance = individual_consciousness.iter()
                .map(|x| (x - mean_consciousness).powi(2))
                .sum::<f32>() / individual_consciousness.len() as f32;
            variance.sqrt()
        };
        
        // High correlation (low std) increases global consciousness
        mean_consciousness * (1.0 - consciousness_std)
    }
    
    fn calculate_trend_consistency(&self, returns: &[f32]) -> f32 {
        if returns.len() < 2 {
            return 0.5;
        }
        
        let mut consistent_periods = 0;
        for i in 1..returns.len() {
            if (returns[i] > 0.0) == (returns[i-1] > 0.0) {
                consistent_periods += 1;
            }
        }
        
        consistent_periods as f32 / (returns.len() - 1) as f32
    }
    
    fn consciousness_aware_loss(&self, features: &Array2<f32>, targets: &Array2<f32>, consciousness: f32) -> f32 {
        // Simplified loss calculation with consciousness weighting
        let mse = ((features.slice(s![.., 3]) - targets.slice(s![.., 0])).mapv(|x| x.powi(2))).mean().unwrap_or(0.0);
        
        // Higher consciousness reduces loss weight (more confident)
        mse * (1.0 - consciousness * 0.3)
    }
    
    fn calculate_prediction_intervals(&self, predictions: &[f32], confidences: &[f32]) -> Vec<(f32, f32)> {
        predictions.iter().zip(confidences.iter())
            .map(|(&pred, &conf)| {
                let margin = pred * (1.0 - conf) * 0.1;  // 10% base margin adjusted by confidence
                (pred - margin, pred + margin)
            })
            .collect()
    }
    
    fn apply_cross_asset_correlation(&self, mut prediction: PredictionResult, global_consciousness: f32) -> PredictionResult {
        // Adjust prediction based on global market consciousness
        let adjustment_factor = 0.8 + global_consciousness * 0.4;  // 0.8 to 1.2 range
        
        prediction.predicted_prices = prediction.predicted_prices.iter()
            .map(|&price| price * adjustment_factor)
            .collect();
        
        prediction.consciousness_state = global_consciousness;
        prediction
    }
    
    fn combine_ensemble_predictions(&self, predictions: &[Vec<f32>]) -> Vec<f32> {
        if predictions.is_empty() {
            return Vec::new();
        }
        
        let horizon = predictions[0].len();
        let mut combined = vec![0.0; horizon];
        
        for i in 0..horizon {
            let sum: f32 = predictions.iter().map(|pred| pred[i]).sum();
            combined[i] = sum / predictions.len() as f32;
        }
        
        combined
    }
    
    fn combine_ensemble_confidences(&self, confidences: &[Vec<f32>]) -> Vec<f32> {
        if confidences.is_empty() {
            return Vec::new();
        }
        
        let horizon = confidences[0].len();
        let mut combined = vec![0.0; horizon];
        
        for i in 0..horizon {
            // Use geometric mean for confidence combination
            let product: f32 = confidences.iter().map(|conf| conf[i]).product();
            combined[i] = product.powf(1.0 / confidences.len() as f32);
        }
        
        combined
    }
    
    fn calculate_sharpe_ratio(&self, returns: &[f32]) -> f32 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f32>() / returns.len() as f32;
        let return_std = {
            let variance = returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f32>() / returns.len() as f32;
            variance.sqrt()
        };
        
        if return_std == 0.0 {
            0.0
        } else {
            mean_return / return_std * (252.0_f32).sqrt()  // Annualized
        }
    }
    
    fn calculate_max_drawdown(&self, prices: &[f32]) -> f32 {
        if prices.is_empty() {
            return 0.0;
        }
        
        let mut max_price = prices[0];
        let mut max_drawdown = 0.0;
        
        for &price in prices.iter().skip(1) {
            if price > max_price {
                max_price = price;
            } else {
                let drawdown = (max_price - price) / max_price;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }
        
        max_drawdown
    }
}

/// Specialized price prediction strategies
pub mod strategies {
    use super::*;
    
    /// Momentum-based price prediction
    pub struct MomentumPredictor {
        base_predictor: PricePredictor,
        momentum_window: usize,
        momentum_threshold: f32,
    }
    
    impl MomentumPredictor {
        pub fn new(lookback: usize, forecast: usize, momentum_window: usize) -> Self {
            Self {
                base_predictor: PricePredictor::new(lookback, forecast),
                momentum_window,
                momentum_threshold: 0.02,  // 2% momentum threshold
            }
        }
        
        pub fn predict_with_momentum(&mut self, symbol: &str, data: &Array2<f32>) -> Result<PredictionResult, String> {
            let mut result = self.base_predictor.predict_single(symbol, data)?;
            
            // Calculate momentum factor
            let momentum = self.calculate_momentum(data);
            
            // Adjust predictions based on momentum
            if momentum.abs() > self.momentum_threshold {
                let momentum_factor = 1.0 + momentum * 0.5;
                result.predicted_prices = result.predicted_prices.iter()
                    .map(|&price| price * momentum_factor)
                    .collect();
            }
            
            Ok(result)
        }
        
        fn calculate_momentum(&self, data: &Array2<f32>) -> f32 {
            let prices = data.slice(s![.., 3]);  // Close prices
            let n = prices.len();
            
            if n < self.momentum_window {
                return 0.0;
            }
            
            let recent_avg = prices.slice(s![n-self.momentum_window..])
                .mean().unwrap_or(0.0);
            let older_avg = prices.slice(s![n-2*self.momentum_window..n-self.momentum_window])
                .mean().unwrap_or(recent_avg);
            
            if older_avg == 0.0 {
                0.0
            } else {
                (recent_avg - older_avg) / older_avg
            }
        }
    }
    
    /// Mean reversion price prediction
    pub struct MeanReversionPredictor {
        base_predictor: PricePredictor,
        reversion_period: usize,
        reversion_strength: f32,
    }
    
    impl MeanReversionPredictor {
        pub fn new(lookback: usize, forecast: usize, reversion_period: usize) -> Self {
            Self {
                base_predictor: PricePredictor::new(lookback, forecast),
                reversion_period,
                reversion_strength: 0.3,
            }
        }
        
        pub fn predict_with_reversion(&mut self, symbol: &str, data: &Array2<f32>) -> Result<PredictionResult, String> {
            let mut result = self.base_predictor.predict_single(symbol, data)?;
            
            // Calculate mean reversion factor
            let reversion_factor = self.calculate_reversion_factor(data);
            
            // Apply mean reversion to predictions
            let mean_price = self.calculate_mean_price(data);
            result.predicted_prices = result.predicted_prices.iter()
                .enumerate()
                .map(|(i, &price)| {
                    let reversion_weight = self.reversion_strength * (1.0 - i as f32 / result.predicted_prices.len() as f32);
                    price * (1.0 - reversion_weight) + mean_price * reversion_weight * reversion_factor
                })
                .collect();
            
            Ok(result)
        }
        
        fn calculate_reversion_factor(&self, data: &Array2<f32>) -> f32 {
            let prices = data.slice(s![.., 3]);  // Close prices
            let current_price = prices[prices.len() - 1];
            let mean_price = self.calculate_mean_price(data);
            
            // Stronger reversion when further from mean
            ((current_price - mean_price) / mean_price).abs()
        }
        
        fn calculate_mean_price(&self, data: &Array2<f32>) -> f32 {
            let prices = data.slice(s![.., 3]);  // Close prices
            let n = prices.len();
            
            if n < self.reversion_period {
                prices.mean().unwrap_or(0.0)
            } else {
                prices.slice(s![n-self.reversion_period..])
                    .mean().unwrap_or(0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_price_predictor_creation() {
        let predictor = PricePredictor::new(60, 10);
        assert_eq!(predictor.lookback_window, 60);
        assert_eq!(predictor.forecast_horizon, 10);
    }
    
    #[test]
    fn test_add_asset() {
        let mut predictor = PricePredictor::new(60, 10);
        predictor.add_asset("AAPL".to_string(), 10);
        assert!(predictor.models.contains_key("AAPL"));
    }
    
    #[test]
    fn test_prediction_intervals() {
        let predictor = PricePredictor::new(60, 10);
        let predictions = vec![100.0, 101.0, 102.0];
        let confidences = vec![0.8, 0.75, 0.9];
        
        let intervals = predictor.calculate_prediction_intervals(&predictions, &confidences);
        assert_eq!(intervals.len(), 3);
        
        // Higher confidence should lead to tighter intervals
        let interval_width_1 = intervals[0].1 - intervals[0].0;
        let interval_width_2 = intervals[2].1 - intervals[2].0;
        assert!(interval_width_2 < interval_width_1);  // Higher confidence = tighter interval
    }
}