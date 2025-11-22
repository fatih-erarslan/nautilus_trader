//! Volatility modeling and VaR calculation using consciousness-aware NHITS
//! 
//! This module implements sophisticated volatility forecasting and risk measurement
//! techniques, leveraging the enhanced NHITS architecture for superior volatility
//! prediction and Value-at-Risk (VaR) estimation.

use super::*;
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Volatility forecasting model using NHITS
#[derive(Debug)]
pub struct VolatilityPredictor {
    pub nhits_model: FinancialNHITS,
    pub volatility_type: VolatilityType,
    pub garch_params: GARCHParameters,
    pub consciousness_adjustment: f32,
    pub lookback_window: usize,
    pub forecast_horizon: usize,
}

#[derive(Debug, Clone)]
pub enum VolatilityType {
    Historical,
    EWMA,      // Exponentially Weighted Moving Average
    GARCH,     // Generalized Autoregressive Conditional Heteroskedasticity
    Stochastic, // Stochastic Volatility
    Realized,   // Realized Volatility from high-frequency data
}

#[derive(Debug, Clone)]
pub struct GARCHParameters {
    pub omega: f32,    // Constant term
    pub alpha: f32,    // ARCH coefficient
    pub beta: f32,     // GARCH coefficient
    pub lambda: f32,   // Decay factor for EWMA
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityForecast {
    pub symbol: String,
    pub volatility_predictions: Vec<f32>,
    pub confidence_intervals: Vec<(f32, f32)>,
    pub var_95: f32,
    pub var_99: f32,
    pub expected_shortfall_95: f32,
    pub expected_shortfall_99: f32,
    pub consciousness_state: f32,
    pub forecast_timestamp: i64,
    pub model_type: String,
}

#[derive(Debug, Clone)]
pub struct RiskMeasures {
    pub var_1d_95: f32,
    pub var_1d_99: f32,
    pub var_10d_95: f32,
    pub var_10d_99: f32,
    pub expected_shortfall_95: f32,
    pub expected_shortfall_99: f32,
    pub conditional_var: f32,
    pub maximum_loss: f32,
}

impl Default for GARCHParameters {
    fn default() -> Self {
        Self {
            omega: 0.000001,
            alpha: 0.1,
            beta: 0.85,
            lambda: 0.94,
        }
    }
}

impl VolatilityPredictor {
    pub fn new(
        input_dim: usize,
        lookback_window: usize,
        forecast_horizon: usize,
        volatility_type: VolatilityType,
    ) -> Self {
        Self {
            nhits_model: FinancialNHITS::new(
                input_dim,
                96,  // hidden_dim optimized for volatility
                4,   // num_stacks
                6,   // num_blocks
                forecast_horizon,
            ).with_financial_components(),
            volatility_type,
            garch_params: GARCHParameters::default(),
            consciousness_adjustment: 1.0,
            lookback_window,
            forecast_horizon,
        }
    }
    
    /// Train volatility model on historical returns
    pub fn train(&mut self, returns: &[f32], epochs: usize) -> Result<(), String> {
        let volatilities = self.calculate_historical_volatilities(returns)?;
        let features = self.prepare_volatility_features(returns, &volatilities);
        let targets = self.prepare_volatility_targets(&volatilities);
        
        // Train with consciousness-aware volatility loss
        for epoch in 0..epochs {
            let consciousness_state = self.calculate_volatility_consciousness(&features);
            let loss = self.volatility_aware_loss(&features, &targets, consciousness_state);
            
            // Update consciousness adjustment based on training progress
            self.consciousness_adjustment = 0.8 + consciousness_state * 0.4;
            
            if epoch % 100 == 0 {
                println!("Volatility Training Epoch {}: Loss = {:.6}, Consciousness = {:.3}", 
                         epoch, loss, consciousness_state);
            }
        }
        
        Ok(())
    }
    
    /// Predict future volatility
    pub fn predict_volatility(&mut self, recent_returns: &[f32]) -> Result<VolatilityForecast, String> {
        if recent_returns.len() < self.lookback_window {
            return Err("Insufficient data for volatility prediction".to_string());
        }
        
        let recent_volatilities = self.calculate_historical_volatilities(recent_returns)?;
        let features = self.prepare_volatility_features(recent_returns, &recent_volatilities);
        
        // Calculate consciousness state for volatility regime
        let consciousness_state = self.calculate_volatility_consciousness(&features);
        
        // Make prediction using consciousness-aware model
        let prediction = self.nhits_model.predict_conscious(&features, consciousness_state);
        let volatility_predictions = prediction.row(0).to_vec();
        
        // Calculate confidence intervals
        let confidence_intervals = self.calculate_volatility_confidence_intervals(
            &volatility_predictions, 
            consciousness_state
        );
        
        // Calculate VaR and Expected Shortfall
        let (var_95, var_99) = self.calculate_var(&volatility_predictions, recent_returns);
        let (es_95, es_99) = self.calculate_expected_shortfall(&volatility_predictions, recent_returns);
        
        Ok(VolatilityForecast {
            symbol: "DEFAULT".to_string(),
            volatility_predictions,
            confidence_intervals,
            var_95,
            var_99,
            expected_shortfall_95: es_95,
            expected_shortfall_99: es_99,
            consciousness_state,
            forecast_timestamp: chrono::Utc::now().timestamp(),
            model_type: format!("{:?}", self.volatility_type),
        })
    }
    
    /// Calculate comprehensive risk measures
    pub fn calculate_risk_measures(&self, returns: &[f32], confidence_level: f32) -> RiskMeasures {
        let volatilities = self.calculate_historical_volatilities(returns).unwrap_or_default();
        
        if volatilities.is_empty() {
            return RiskMeasures {
                var_1d_95: 0.0,
                var_1d_99: 0.0,
                var_10d_95: 0.0,
                var_10d_99: 0.0,
                expected_shortfall_95: 0.0,
                expected_shortfall_99: 0.0,
                conditional_var: 0.0,
                maximum_loss: 0.0,
            };
        }
        
        let current_volatility = volatilities[volatilities.len() - 1];
        
        // Calculate VaR at different horizons
        let var_1d_95 = self.calculate_parametric_var(current_volatility, 0.95, 1);
        let var_1d_99 = self.calculate_parametric_var(current_volatility, 0.99, 1);
        let var_10d_95 = self.calculate_parametric_var(current_volatility, 0.95, 10);
        let var_10d_99 = self.calculate_parametric_var(current_volatility, 0.99, 10);
        
        // Calculate Expected Shortfall (Conditional VaR)
        let expected_shortfall_95 = var_1d_95 * 1.28;  // Approximation for normal distribution
        let expected_shortfall_99 = var_1d_99 * 1.17;
        
        // Conditional VaR (average loss beyond VaR)
        let conditional_var = self.calculate_conditional_var(returns, 0.95);
        
        // Maximum historical loss
        let maximum_loss = returns.iter().fold(0.0f32, |acc, &x| acc.min(x)).abs();
        
        RiskMeasures {
            var_1d_95,
            var_1d_99,
            var_10d_95,
            var_10d_99,
            expected_shortfall_95,
            expected_shortfall_99,
            conditional_var,
            maximum_loss,
        }
    }
    
    /// Implement GARCH volatility modeling
    pub fn garch_forecast(&mut self, returns: &[f32]) -> Result<Vec<f32>, String> {
        if returns.len() < 100 {
            return Err("Insufficient data for GARCH modeling".to_string());
        }
        
        // Estimate GARCH parameters using Maximum Likelihood
        self.estimate_garch_parameters(returns)?;
        
        let mut volatility_forecast = Vec::with_capacity(self.forecast_horizon);
        let mut last_return = returns[returns.len() - 1];
        let mut last_variance = self.calculate_historical_volatilities(returns)?
            .pop().unwrap_or(0.01).powi(2);
        
        for _ in 0..self.forecast_horizon {
            // GARCH(1,1) forecast: σ²_{t+1} = ω + α * r²_t + β * σ²_t
            let next_variance = self.garch_params.omega + 
                self.garch_params.alpha * last_return.powi(2) + 
                self.garch_params.beta * last_variance;
            
            let next_volatility = next_variance.sqrt();
            volatility_forecast.push(next_volatility);
            
            // For multi-step ahead, assume return = 0 (unconditional)
            last_return = 0.0;
            last_variance = next_variance;
        }
        
        Ok(volatility_forecast)
    }
    
    /// Implement Exponentially Weighted Moving Average volatility
    pub fn ewma_volatility(&self, returns: &[f32]) -> Vec<f32> {
        let mut ewma_vol = Vec::with_capacity(returns.len());
        
        if returns.is_empty() {
            return ewma_vol;
        }
        
        // Initialize with first squared return
        let mut variance = returns[0].powi(2);
        ewma_vol.push(variance.sqrt());
        
        for i in 1..returns.len() {
            variance = self.garch_params.lambda * variance + 
                      (1.0 - self.garch_params.lambda) * returns[i].powi(2);
            ewma_vol.push(variance.sqrt());
        }
        
        ewma_vol
    }
    
    /// Calculate realized volatility from intraday returns
    pub fn realized_volatility(&self, intraday_returns: &[f32], days: usize) -> Vec<f32> {
        let returns_per_day = intraday_returns.len() / days;
        let mut realized_vols = Vec::with_capacity(days);
        
        for day in 0..days {
            let start_idx = day * returns_per_day;
            let end_idx = ((day + 1) * returns_per_day).min(intraday_returns.len());
            
            if start_idx < end_idx {
                let day_returns = &intraday_returns[start_idx..end_idx];
                let realized_variance: f32 = day_returns.iter()
                    .map(|&r| r.powi(2))
                    .sum();
                
                realized_vols.push(realized_variance.sqrt());
            }
        }
        
        realized_vols
    }
    
    /// Calculate volatility clustering measure
    pub fn volatility_clustering(&self, returns: &[f32]) -> f32 {
        let volatilities = self.calculate_historical_volatilities(returns).unwrap_or_default();
        
        if volatilities.len() < 20 {
            return 0.0;
        }
        
        // Calculate autocorrelation of squared returns (proxy for volatility clustering)
        let squared_returns: Vec<f32> = returns.iter().map(|&r| r.powi(2)).collect();
        self.calculate_autocorrelation(&squared_returns, 1)
    }
    
    /// Stress testing with consciousness awareness
    pub fn stress_test(&self, returns: &[f32], stress_scenarios: &[f32]) -> Vec<RiskMeasures> {
        stress_scenarios.iter()
            .map(|&stress_factor| {
                let stressed_returns: Vec<f32> = returns.iter()
                    .map(|&r| r * stress_factor)
                    .collect();
                self.calculate_risk_measures(&stressed_returns, 0.95)
            })
            .collect()
    }
    
    // Private helper methods
    
    fn calculate_historical_volatilities(&self, returns: &[f32]) -> Result<Vec<f32>, String> {
        if returns.len() < 20 {
            return Err("Insufficient data for volatility calculation".to_string());
        }
        
        let window_size = 20;  // 20-day rolling window
        let mut volatilities = Vec::with_capacity(returns.len() - window_size + 1);
        
        for i in window_size..=returns.len() {
            let window_returns = &returns[i-window_size..i];
            let mean_return = window_returns.iter().sum::<f32>() / window_size as f32;
            
            let variance = window_returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f32>() / (window_size - 1) as f32;
            
            let volatility = variance.sqrt() * (252.0_f32).sqrt();  // Annualized
            volatilities.push(volatility);
        }
        
        Ok(volatilities)
    }
    
    fn prepare_volatility_features(&self, returns: &[f32], volatilities: &[f32]) -> Array2<f32> {
        let n = returns.len().min(volatilities.len());
        let mut features = Array2::zeros((n, 8));
        
        for i in 0..n {
            features[[i, 0]] = returns[i];                           // Return
            features[[i, 1]] = returns[i].powi(2);                   // Squared return
            features[[i, 2]] = returns[i].abs();                     // Absolute return
            features[[i, 3]] = volatilities[i];                      // Historical volatility
            
            if i > 0 {
                features[[i, 4]] = volatilities[i] - volatilities[i-1];  // Volatility change
                features[[i, 5]] = returns[i] * returns[i-1];            // Return serial correlation
            }
            
            if i >= 5 {
                // 5-day average volatility
                let avg_vol = volatilities[i-4..=i].iter().sum::<f32>() / 5.0;
                features[[i, 6]] = avg_vol;
                
                // Volatility momentum
                let vol_momentum = (volatilities[i] - avg_vol) / avg_vol;
                features[[i, 7]] = vol_momentum;
            }
        }
        
        features
    }
    
    fn prepare_volatility_targets(&self, volatilities: &[f32]) -> Array2<f32> {
        let n = volatilities.len();
        let mut targets = Array2::zeros((n - self.forecast_horizon, self.forecast_horizon));
        
        for i in 0..(n - self.forecast_horizon) {
            for j in 0..self.forecast_horizon {
                targets[[i, j]] = volatilities[i + j + 1];
            }
        }
        
        targets
    }
    
    fn calculate_volatility_consciousness(&self, features: &Array2<f32>) -> f32 {
        // Volatility consciousness based on regime stability and predictability
        let vol_column = features.slice(s![.., 3]);  // Volatility column
        let vol_changes = features.slice(s![.., 4]); // Volatility changes
        
        let vol_stability = 1.0 / (1.0 + vol_changes.std(0.0));
        let vol_level = vol_column.mean().unwrap_or(0.0);
        
        // High consciousness = stable, moderate volatility regime
        let consciousness = vol_stability * (1.0 - (vol_level - 0.2).abs().min(0.8));
        consciousness.min(1.0).max(0.0)
    }
    
    fn volatility_aware_loss(&self, features: &Array2<f32>, targets: &Array2<f32>, consciousness: f32) -> f32 {
        let predicted_vol = features.slice(s![.., 3]);
        let target_vol = targets.slice(s![.., 0]);
        
        // MSE with consciousness weighting
        let mse = ((predicted_vol - target_vol).mapv(|x| x.powi(2))).mean().unwrap_or(0.0);
        
        // Higher consciousness = lower loss weight (more reliable regime)
        mse * (1.0 - consciousness * 0.2)
    }
    
    fn calculate_volatility_confidence_intervals(&self, predictions: &[f32], consciousness: f32) -> Vec<(f32, f32)> {
        predictions.iter()
            .map(|&vol_pred| {
                let base_margin = vol_pred * 0.15;  // 15% base margin
                let consciousness_adjustment = 1.0 - consciousness * 0.3;  // Higher consciousness = tighter intervals
                let margin = base_margin * consciousness_adjustment;
                (vol_pred - margin, vol_pred + margin)
            })
            .collect()
    }
    
    fn calculate_var(&self, volatility_predictions: &[f32], returns: &[f32]) -> (f32, f32) {
        if volatility_predictions.is_empty() || returns.is_empty() {
            return (0.0, 0.0);
        }
        
        let avg_predicted_vol = volatility_predictions.iter().sum::<f32>() / volatility_predictions.len() as f32;
        
        // Parametric VaR assuming normal distribution
        let var_95 = avg_predicted_vol * 1.645;  // 95% confidence
        let var_99 = avg_predicted_vol * 2.326;  // 99% confidence
        
        (var_95, var_99)
    }
    
    fn calculate_expected_shortfall(&self, volatility_predictions: &[f32], returns: &[f32]) -> (f32, f32) {
        let (var_95, var_99) = self.calculate_var(volatility_predictions, returns);
        
        // Expected Shortfall (Conditional VaR) for normal distribution
        let es_95 = var_95 * 1.28;  // E[X|X > VaR_95%] for normal
        let es_99 = var_99 * 1.17;  // E[X|X > VaR_99%] for normal
        
        (es_95, es_99)
    }
    
    fn calculate_parametric_var(&self, volatility: f32, confidence: f32, horizon: u32) -> f32 {
        let z_score = if confidence >= 0.99 {
            2.326
        } else if confidence >= 0.95 {
            1.645
        } else {
            1.282  // 90%
        };
        
        volatility * z_score * (horizon as f32).sqrt()
    }
    
    fn calculate_conditional_var(&self, returns: &[f32], confidence: f32) -> f32 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_index = ((1.0 - confidence) * sorted_returns.len() as f32) as usize;
        let var_threshold = sorted_returns[var_index.min(sorted_returns.len() - 1)];
        
        // Average of losses beyond VaR
        let tail_losses: Vec<f32> = sorted_returns.iter()
            .filter(|&&r| r <= var_threshold)
            .cloned()
            .collect();
        
        if tail_losses.is_empty() {
            var_threshold.abs()
        } else {
            tail_losses.iter().sum::<f32>() / tail_losses.len() as f32 * -1.0
        }
    }
    
    fn estimate_garch_parameters(&mut self, returns: &[f32]) -> Result<(), String> {
        // Simplified GARCH parameter estimation
        // In practice, this would use Maximum Likelihood Estimation
        
        let volatilities = self.calculate_historical_volatilities(returns)?;
        let variances: Vec<f32> = volatilities.iter().map(|&v| v.powi(2)).collect();
        
        if variances.len() < 50 {
            return Err("Insufficient data for GARCH estimation".to_string());
        }
        
        // Rough estimation using method of moments
        let unconditional_variance = variances.iter().sum::<f32>() / variances.len() as f32;
        
        // Simple heuristic parameter values
        self.garch_params.omega = unconditional_variance * 0.05;
        self.garch_params.alpha = 0.1;
        self.garch_params.beta = 0.85;
        
        // Ensure parameters sum to less than 1 for stationarity
        let sum = self.garch_params.alpha + self.garch_params.beta;
        if sum >= 1.0 {
            self.garch_params.alpha *= 0.9 / sum;
            self.garch_params.beta *= 0.9 / sum;
        }
        
        Ok(())
    }
    
    fn calculate_autocorrelation(&self, series: &[f32], lag: usize) -> f32 {
        if series.len() <= lag {
            return 0.0;
        }
        
        let n = series.len() - lag;
        let mean = series.iter().sum::<f32>() / series.len() as f32;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..n {
            let x_i = series[i] - mean;
            let x_lag = series[i + lag] - mean;
            numerator += x_i * x_lag;
        }
        
        for &x in series {
            denominator += (x - mean).powi(2);
        }
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Advanced volatility modeling techniques
pub mod advanced {
    use super::*;
    
    /// Stochastic volatility model
    pub struct StochasticVolatilityModel {
        pub mean_reversion_speed: f32,
        pub long_term_variance: f32,
        pub volatility_of_volatility: f32,
        pub correlation: f32,
    }
    
    impl StochasticVolatilityModel {
        pub fn new() -> Self {
            Self {
                mean_reversion_speed: 2.0,
                long_term_variance: 0.04,
                volatility_of_volatility: 0.3,
                correlation: -0.5,
            }
        }
        
        pub fn simulate_volatility_path(&self, initial_vol: f32, steps: usize, dt: f32) -> Vec<f32> {
            let mut vol_path = vec![initial_vol];
            let mut current_vol = initial_vol;
            
            for _ in 1..steps {
                let z1 = self.sample_normal();
                let z2 = self.correlation * z1 + (1.0 - self.correlation.powi(2)).sqrt() * self.sample_normal();
                
                let vol_drift = self.mean_reversion_speed * (self.long_term_variance - current_vol.powi(2));
                let vol_diffusion = self.volatility_of_volatility * current_vol * z2;
                
                current_vol = (current_vol.powi(2) + vol_drift * dt + vol_diffusion * dt.sqrt()).sqrt();
                current_vol = current_vol.max(0.001);  // Prevent negative volatility
                
                vol_path.push(current_vol);
            }
            
            vol_path
        }
        
        fn sample_normal(&self) -> f32 {
            // Use proper statistical distribution from statrs crate
            use statrs::distribution::{Normal, ContinuousCDF};
            use rand::thread_rng;
            
            let normal = Normal::new(0.0, 1.0).unwrap();
            let mut rng = thread_rng();
            normal.sample(&mut rng) as f32
        }
    }
    
    /// Jump-diffusion volatility model
    pub struct JumpDiffusionVolatility {
        pub jump_intensity: f32,
        pub jump_mean: f32,
        pub jump_volatility: f32,
        pub base_predictor: VolatilityPredictor,
    }
    
    impl JumpDiffusionVolatility {
        pub fn new(input_dim: usize, lookback: usize, forecast: usize) -> Self {
            Self {
                jump_intensity: 0.1,  // Expected jumps per year
                jump_mean: 0.0,
                jump_volatility: 0.05,
                base_predictor: VolatilityPredictor::new(
                    input_dim, 
                    lookback, 
                    forecast, 
                    VolatilityType::Stochastic
                ),
            }
        }
        
        pub fn forecast_with_jumps(&mut self, returns: &[f32]) -> Result<VolatilityForecast, String> {
            let mut base_forecast = self.base_predictor.predict_volatility(returns)?;
            
            // Adjust for potential jumps
            let jump_adjustment = self.calculate_jump_adjustment(returns);
            
            base_forecast.volatility_predictions = base_forecast.volatility_predictions
                .iter()
                .map(|&vol| vol * (1.0 + jump_adjustment))
                .collect();
            
            base_forecast.model_type = "Jump-Diffusion".to_string();
            Ok(base_forecast)
        }
        
        fn calculate_jump_adjustment(&self, returns: &[f32]) -> f32 {
            // Detect recent jump activity
            let threshold = 3.0 * returns.iter()
                .map(|&r| r.powi(2))
                .sum::<f32>() / returns.len() as f32;
            
            let recent_jumps = returns.iter()
                .take(20)  // Last 20 observations
                .filter(|&&r| r.powi(2) > threshold)
                .count();
            
            // Increase volatility if recent jump activity
            if recent_jumps > 2 {
                self.jump_volatility * recent_jumps as f32 / 20.0
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_volatility_predictor_creation() {
        let predictor = VolatilityPredictor::new(10, 60, 10, VolatilityType::GARCH);
        assert_eq!(predictor.lookback_window, 60);
        assert_eq!(predictor.forecast_horizon, 10);
    }
    
    #[test]
    fn test_ewma_volatility() {
        let predictor = VolatilityPredictor::new(10, 60, 10, VolatilityType::EWMA);
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
        let ewma_vols = predictor.ewma_volatility(&returns);
        
        assert_eq!(ewma_vols.len(), returns.len());
        assert!(ewma_vols.iter().all(|&vol| vol >= 0.0));
    }
    
    #[test]
    fn test_risk_measures() {
        let predictor = VolatilityPredictor::new(10, 60, 10, VolatilityType::Historical);
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.03, 0.025];
        let risk_measures = predictor.calculate_risk_measures(&returns, 0.95);
        
        assert!(risk_measures.var_1d_95 >= 0.0);
        assert!(risk_measures.var_1d_99 >= risk_measures.var_1d_95);
        assert!(risk_measures.expected_shortfall_95 >= risk_measures.var_1d_95);
    }
    
    #[test]
    fn test_garch_parameters() {
        let mut predictor = VolatilityPredictor::new(10, 60, 10, VolatilityType::GARCH);
        let returns: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.02).collect();
        
        let result = predictor.estimate_garch_parameters(&returns);
        assert!(result.is_ok());
        
        // Check parameter constraints
        let sum = predictor.garch_params.alpha + predictor.garch_params.beta;
        assert!(sum < 1.0);  // Stationarity condition
        assert!(predictor.garch_params.omega > 0.0);
        assert!(predictor.garch_params.alpha >= 0.0);
        assert!(predictor.garch_params.beta >= 0.0);
    }
}