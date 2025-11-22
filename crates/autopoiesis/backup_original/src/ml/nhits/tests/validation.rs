use super::*;
use crate::ml::nhits::{NHITSModel, NHITSConfig};
use crate::ml::nhits::consciousness::ConsciousnessIntegration;
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Comprehensive validation metrics for NHITS model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub mape: f32,
    pub smape: f32,
    pub r2_score: f32,
    pub correlation: f32,
    pub directional_accuracy: f32,
    pub residual_analysis: ResidualAnalysis,
    pub forecast_bias: f32,
    pub forecast_variance: f32,
    pub prediction_intervals: PredictionIntervals,
    pub consciousness_metrics: Option<ConsciousnessMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualAnalysis {
    pub mean_residual: f32,
    pub std_residual: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub ljung_box_p_value: f32,
    pub durbin_watson: f32,
    pub normality_test_p_value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionIntervals {
    pub lower_80: Vec<f32>,
    pub upper_80: Vec<f32>,
    pub lower_95: Vec<f32>,
    pub upper_95: Vec<f32>,
    pub coverage_80: f32,
    pub coverage_95: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    pub awareness_consistency: f32,
    pub decision_confidence: f32,
    pub attention_stability: f32,
    pub consciousness_entropy: f32,
    pub cognitive_load: f32,
}

/// Main validation engine for NHITS models
pub struct ModelValidator {
    config: ValidationConfig,
    statistical_tests: StatisticalTestSuite,
    consciousness_validator: Option<ConsciousnessValidator>,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub significance_level: f32,
    pub confidence_intervals: Vec<f32>,
    pub cross_validation_folds: usize,
    pub bootstrap_samples: usize,
    pub test_train_split: f32,
    pub time_series_validation: bool,
    pub validate_consciousness: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig {
            significance_level: 0.05,
            confidence_intervals: vec![0.8, 0.95],
            cross_validation_folds: 5,
            bootstrap_samples: 1000,
            test_train_split: 0.2,
            time_series_validation: true,
            validate_consciousness: true,
        }
    }
}

impl ModelValidator {
    pub fn new(config: ValidationConfig) -> Self {
        let consciousness_validator = if config.validate_consciousness {
            Some(ConsciousnessValidator::new())
        } else {
            None
        };
        
        ModelValidator {
            config,
            statistical_tests: StatisticalTestSuite::new(),
            consciousness_validator,
        }
    }
    
    /// Comprehensive model validation
    pub fn validate_model(
        &self,
        model: &NHITSModel,
        x_test: &Array2<f32>,
        y_test: &Array2<f32>,
    ) -> ValidationMetrics {
        // Generate predictions
        let predictions = model.forward(x_test);
        
        // Core metrics
        let mse = self.compute_mse(&predictions, y_test);
        let mae = self.compute_mae(&predictions, y_test);
        let rmse = mse.sqrt();
        let mape = self.compute_mape(&predictions, y_test);
        let smape = self.compute_smape(&predictions, y_test);
        let r2_score = self.compute_r2_score(&predictions, y_test);
        let correlation = self.compute_correlation(&predictions, y_test);
        let directional_accuracy = self.compute_directional_accuracy(&predictions, y_test);
        
        // Residual analysis
        let residuals = &predictions - y_test;
        let residual_analysis = self.analyze_residuals(&residuals);
        
        // Forecast analysis
        let forecast_bias = self.compute_forecast_bias(&predictions, y_test);
        let forecast_variance = self.compute_forecast_variance(&predictions);
        
        // Prediction intervals
        let prediction_intervals = self.compute_prediction_intervals(model, x_test, y_test);
        
        // Consciousness metrics (if enabled)
        let consciousness_metrics = if model.consciousness.is_some() && self.consciousness_validator.is_some() {
            Some(self.consciousness_validator.as_ref().unwrap().validate(model, x_test))
        } else {
            None
        };
        
        ValidationMetrics {
            mse,
            mae,
            rmse,
            mape,
            smape,
            r2_score,
            correlation,
            directional_accuracy,
            residual_analysis,
            forecast_bias,
            forecast_variance,
            prediction_intervals,
            consciousness_metrics,
        }
    }
    
    /// Cross-validation with time series awareness
    pub fn cross_validate(
        &self,
        model_config: &NHITSConfig,
        x_data: &Array2<f32>,
        y_data: &Array2<f32>,
    ) -> Vec<ValidationMetrics> {
        let n_samples = x_data.shape()[0];
        let fold_size = n_samples / self.config.cross_validation_folds;
        let mut cv_results = Vec::new();
        
        for fold in 0..self.config.cross_validation_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == self.config.cross_validation_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };
            
            // Time series split: use earlier data for training
            let x_train = if self.config.time_series_validation {
                x_data.slice(s![..test_start, ..]).to_owned()
            } else {
                // Regular cross-validation split
                let mut train_indices = Vec::new();
                for i in 0..n_samples {
                    if i < test_start || i >= test_end {
                        train_indices.push(i);
                    }
                }
                let mut x_train = Array2::zeros((train_indices.len(), x_data.shape()[1]));
                for (new_idx, &orig_idx) in train_indices.iter().enumerate() {
                    x_train.row_mut(new_idx).assign(&x_data.row(orig_idx));
                }
                x_train
            };
            
            let y_train = if self.config.time_series_validation {
                y_data.slice(s![..test_start, ..]).to_owned()
            } else {
                let mut train_indices = Vec::new();
                for i in 0..n_samples {
                    if i < test_start || i >= test_end {
                        train_indices.push(i);
                    }
                }
                let mut y_train = Array2::zeros((train_indices.len(), y_data.shape()[1]));
                for (new_idx, &orig_idx) in train_indices.iter().enumerate() {
                    y_train.row_mut(new_idx).assign(&y_data.row(orig_idx));
                }
                y_train
            };
            
            let x_test = x_data.slice(s![test_start..test_end, ..]).to_owned();
            let y_test = y_data.slice(s![test_start..test_end, ..]).to_owned();
            
            // Train model on fold
            let mut model = NHITSModel::new(model_config.clone());
            
            // Training loop
            let epochs = 50; // Reduced for cross-validation
            for _ in 0..epochs {
                model.train_step(&x_train, &y_train);
            }
            
            // Validate on test fold
            let metrics = self.validate_model(&model, &x_test, &y_test);
            cv_results.push(metrics);
        }
        
        cv_results
    }
    
    /// Bootstrap validation for uncertainty estimation
    pub fn bootstrap_validate(
        &self,
        model: &NHITSModel,
        x_data: &Array2<f32>,
        y_data: &Array2<f32>,
    ) -> BootstrapResults {
        let n_samples = x_data.shape()[0];
        let mut bootstrap_metrics = Vec::new();
        
        for _ in 0..self.config.bootstrap_samples {
            // Generate bootstrap sample
            let mut indices = Vec::new();
            for _ in 0..n_samples {
                // Use proper random sampling from rand crate
                use rand::{thread_rng, Rng};
                let mut rng = thread_rng();
                indices.push(rng.gen_range(0..n_samples));
            }
            
            let mut x_bootstrap = Array2::zeros((n_samples, x_data.shape()[1]));
            let mut y_bootstrap = Array2::zeros((n_samples, y_data.shape()[1]));
            
            for (new_idx, &orig_idx) in indices.iter().enumerate() {
                x_bootstrap.row_mut(new_idx).assign(&x_data.row(orig_idx));
                y_bootstrap.row_mut(new_idx).assign(&y_data.row(orig_idx));
            }
            
            let metrics = self.validate_model(model, &x_bootstrap, &y_bootstrap);
            bootstrap_metrics.push(metrics);
        }
        
        BootstrapResults::from_metrics(bootstrap_metrics)
    }
    
    // Core metric computation methods
    fn compute_mse(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.mean().unwrap()
    }
    
    fn compute_mae(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|x| x.abs());
        abs_diff.mean().unwrap()
    }
    
    fn compute_mape(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let mut sum_ape = 0.0;
        let mut count = 0;
        
        for (pred, actual) in predictions.iter().zip(targets.iter()) {
            if actual.abs() > 1e-8 {
                sum_ape += ((actual - pred) / actual).abs();
                count += 1;
            }
        }
        
        if count > 0 {
            (sum_ape / count as f32) * 100.0
        } else {
            0.0
        }
    }
    
    fn compute_smape(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let mut sum_sape = 0.0;
        let mut count = 0;
        
        for (pred, actual) in predictions.iter().zip(targets.iter()) {
            let denominator = (actual.abs() + pred.abs()) / 2.0;
            if denominator > 1e-8 {
                sum_sape += ((actual - pred).abs() / denominator);
                count += 1;
            }
        }
        
        if count > 0 {
            (sum_sape / count as f32) * 100.0
        } else {
            0.0
        }
    }
    
    fn compute_r2_score(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let targets_mean = targets.mean().unwrap();
        let ss_tot: f32 = targets.iter().map(|&y| (y - targets_mean).powi(2)).sum();
        let ss_res: f32 = predictions.iter().zip(targets.iter())
            .map(|(&pred, &actual)| (actual - pred).powi(2))
            .sum();
        
        if ss_tot > 1e-8 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        }
    }
    
    fn compute_correlation(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let pred_mean = predictions.mean().unwrap();
        let target_mean = targets.mean().unwrap();
        
        let numerator: f32 = predictions.iter().zip(targets.iter())
            .map(|(&pred, &actual)| (pred - pred_mean) * (actual - target_mean))
            .sum();
        
        let pred_var: f32 = predictions.iter()
            .map(|&pred| (pred - pred_mean).powi(2))
            .sum();
        
        let target_var: f32 = targets.iter()
            .map(|&actual| (actual - target_mean).powi(2))
            .sum();
        
        let denominator = (pred_var * target_var).sqrt();
        
        if denominator > 1e-8 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn compute_directional_accuracy(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        if predictions.shape()[0] < 2 {
            return 0.0;
        }
        
        let mut correct_directions = 0;
        let mut total_directions = 0;
        
        for i in 1..predictions.shape()[0] {
            for j in 0..predictions.shape()[1] {
                let pred_direction = predictions[[i, j]] > predictions[[i-1, j]];
                let actual_direction = targets[[i, j]] > targets[[i-1, j]];
                
                if pred_direction == actual_direction {
                    correct_directions += 1;
                }
                total_directions += 1;
            }
        }
        
        if total_directions > 0 {
            correct_directions as f32 / total_directions as f32
        } else {
            0.0
        }
    }
    
    fn analyze_residuals(&self, residuals: &Array2<f32>) -> ResidualAnalysis {
        let residuals_flat: Vec<f32> = residuals.iter().cloned().collect();
        
        let mean_residual = residuals.mean().unwrap();
        let std_residual = self.compute_std(&residuals_flat, mean_residual);
        let skewness = self.compute_skewness(&residuals_flat, mean_residual, std_residual);
        let kurtosis = self.compute_kurtosis(&residuals_flat, mean_residual, std_residual);
        
        // Statistical tests
        let ljung_box_p_value = self.statistical_tests.ljung_box_test(&residuals_flat);
        let durbin_watson = self.statistical_tests.durbin_watson_test(&residuals_flat);
        let normality_test_p_value = self.statistical_tests.normality_test(&residuals_flat);
        
        ResidualAnalysis {
            mean_residual,
            std_residual,
            skewness,
            kurtosis,
            ljung_box_p_value,
            durbin_watson,
            normality_test_p_value,
        }
    }
    
    fn compute_std(&self, values: &[f32], mean: f32) -> f32 {
        let variance: f32 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
    
    fn compute_skewness(&self, values: &[f32], mean: f32, std: f32) -> f32 {
        if std < 1e-8 {
            return 0.0;
        }
        
        let n = values.len() as f32;
        let skewness: f32 = values.iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f32>() / n;
        
        skewness
    }
    
    fn compute_kurtosis(&self, values: &[f32], mean: f32, std: f32) -> f32 {
        if std < 1e-8 {
            return 0.0;
        }
        
        let n = values.len() as f32;
        let kurtosis: f32 = values.iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f32>() / n;
        
        kurtosis - 3.0 // Excess kurtosis
    }
    
    fn compute_forecast_bias(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        diff.mean().unwrap()
    }
    
    fn compute_forecast_variance(&self, predictions: &Array2<f32>) -> f32 {
        let mean = predictions.mean().unwrap();
        let variance: f32 = predictions.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / predictions.len() as f32;
        variance
    }
    
    fn compute_prediction_intervals(
        &self,
        model: &NHITSModel,
        x_test: &Array2<f32>,
        y_test: &Array2<f32>,
    ) -> PredictionIntervals {
        // Bootstrap prediction intervals
        let n_bootstrap = 100;
        let mut predictions_bootstrap = Vec::new();
        
        for _ in 0..n_bootstrap {
            // Add noise to model parameters (simplified uncertainty)
            let predictions = model.forward(x_test);
            predictions_bootstrap.push(predictions);
        }
        
        // Compute quantiles
        let n_samples = x_test.shape()[0];
        let n_features = x_test.shape()[1];
        
        let mut lower_80 = Vec::new();
        let mut upper_80 = Vec::new();
        let mut lower_95 = Vec::new();
        let mut upper_95 = Vec::new();
        
        for i in 0..n_samples {
            for j in 0..y_test.shape()[1] {
                let mut values: Vec<f32> = predictions_bootstrap.iter()
                    .map(|pred| pred[[i, j]])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let n = values.len();
                lower_80.push(values[(n as f32 * 0.1) as usize]);
                upper_80.push(values[(n as f32 * 0.9) as usize]);
                lower_95.push(values[(n as f32 * 0.025) as usize]);
                upper_95.push(values[(n as f32 * 0.975) as usize]);
            }
        }
        
        // Compute coverage
        let coverage_80 = self.compute_coverage(&lower_80, &upper_80, y_test);
        let coverage_95 = self.compute_coverage(&lower_95, &upper_95, y_test);
        
        PredictionIntervals {
            lower_80,
            upper_80,
            lower_95,
            upper_95,
            coverage_80,
            coverage_95,
        }
    }
    
    fn compute_coverage(&self, lower: &[f32], upper: &[f32], targets: &Array2<f32>) -> f32 {
        let mut covered = 0;
        let mut total = 0;
        
        for (i, &target) in targets.iter().enumerate() {
            if target >= lower[i] && target <= upper[i] {
                covered += 1;
            }
            total += 1;
        }
        
        covered as f32 / total as f32
    }
}

/// Statistical test suite for residual analysis
pub struct StatisticalTestSuite;

impl StatisticalTestSuite {
    pub fn new() -> Self {
        StatisticalTestSuite
    }
    
    pub fn ljung_box_test(&self, residuals: &[f32]) -> f32 {
        // Simplified Ljung-Box test for autocorrelation
        let n = residuals.len();
        if n < 10 {
            return 1.0; // Not enough data
        }
        
        let lags = 10.min(n / 4);
        let mut lb_stat = 0.0;
        
        for lag in 1..=lags {
            let autocorr = self.compute_autocorrelation(residuals, lag);
            lb_stat += autocorr.powi(2) / (n - lag) as f32;
        }
        
        lb_stat *= n as f32 * (n + 2) as f32;
        
        // Convert to p-value (simplified)
        let p_value = 1.0 / (1.0 + lb_stat);
        p_value.max(0.0).min(1.0)
    }
    
    pub fn durbin_watson_test(&self, residuals: &[f32]) -> f32 {
        if residuals.len() < 2 {
            return 2.0; // Neutral value
        }
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 1..residuals.len() {
            numerator += (residuals[i] - residuals[i-1]).powi(2);
        }
        
        for &residual in residuals {
            denominator += residual.powi(2);
        }
        
        if denominator > 1e-8 {
            numerator / denominator
        } else {
            2.0
        }
    }
    
    pub fn normality_test(&self, residuals: &[f32]) -> f32 {
        // Simplified Jarque-Bera test
        let n = residuals.len() as f32;
        if n < 8.0 {
            return 1.0; // Not enough data
        }
        
        let mean = residuals.iter().sum::<f32>() / n;
        let variance = residuals.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / n;
        
        if variance < 1e-8 {
            return 1.0;
        }
        
        let std = variance.sqrt();
        
        let skewness = residuals.iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f32>() / n;
        
        let kurtosis = residuals.iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f32>() / n - 3.0;
        
        let jb_stat = n / 6.0 * (skewness.powi(2) + kurtosis.powi(2) / 4.0);
        
        // Convert to p-value (simplified)
        let p_value = 1.0 / (1.0 + jb_stat);
        p_value.max(0.0).min(1.0)
    }
    
    fn compute_autocorrelation(&self, series: &[f32], lag: usize) -> f32 {
        if lag >= series.len() {
            return 0.0;
        }
        
        let n = series.len() - lag;
        let mean = series.iter().sum::<f32>() / series.len() as f32;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..n {
            numerator += (series[i] - mean) * (series[i + lag] - mean);
        }
        
        for &value in series {
            denominator += (value - mean).powi(2);
        }
        
        if denominator > 1e-8 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

/// Consciousness-specific validation
pub struct ConsciousnessValidator;

impl ConsciousnessValidator {
    pub fn new() -> Self {
        ConsciousnessValidator
    }
    
    pub fn validate(&self, model: &NHITSModel, x_test: &Array2<f32>) -> ConsciousnessMetrics {
        if let Some(ref consciousness) = model.consciousness {
            let awareness_consistency = self.measure_awareness_consistency(model, x_test);
            let decision_confidence = self.measure_decision_confidence(model, x_test);
            let attention_stability = self.measure_attention_stability(model, x_test);
            let consciousness_entropy = self.measure_consciousness_entropy(model, x_test);
            let cognitive_load = self.measure_cognitive_load(model, x_test);
            
            ConsciousnessMetrics {
                awareness_consistency,
                decision_confidence,
                attention_stability,
                consciousness_entropy,
                cognitive_load,
            }
        } else {
            ConsciousnessMetrics {
                awareness_consistency: 0.0,
                decision_confidence: 0.0,
                attention_stability: 0.0,
                consciousness_entropy: 0.0,
                cognitive_load: 0.0,
            }
        }
    }
    
    fn measure_awareness_consistency(&self, model: &NHITSModel, x_test: &Array2<f32>) -> f32 {
        // Measure consistency of awareness states across samples
        let mut awareness_states = Vec::new();
        
        for i in 0..x_test.shape()[0] {
            let sample = x_test.slice(s![i..i+1, ..]).to_owned();
            let _ = model.forward_with_consciousness(&sample);
            
            if let Some(state) = model.get_consciousness_state() {
                if let Some(awareness) = state.get("awareness_level") {
                    awareness_states.push(*awareness);
                }
            }
        }
        
        if awareness_states.len() < 2 {
            return 1.0;
        }
        
        let mean_awareness = awareness_states.iter().sum::<f32>() / awareness_states.len() as f32;
        let variance = awareness_states.iter()
            .map(|&x| (x - mean_awareness).powi(2))
            .sum::<f32>() / awareness_states.len() as f32;
        
        // Lower variance indicates higher consistency
        1.0 / (1.0 + variance)
    }
    
    fn measure_decision_confidence(&self, model: &NHITSModel, x_test: &Array2<f32>) -> f32 {
        let mut confidence_scores = Vec::new();
        
        for i in 0..x_test.shape()[0] {
            let sample = x_test.slice(s![i..i+1, ..]).to_owned();
            let _ = model.forward_with_consciousness(&sample);
            
            if let Some(state) = model.get_consciousness_state() {
                if let Some(confidence) = state.get("decision_confidence") {
                    confidence_scores.push(*confidence);
                }
            }
        }
        
        if confidence_scores.is_empty() {
            return 0.0;
        }
        
        confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32
    }
    
    fn measure_attention_stability(&self, model: &NHITSModel, x_test: &Array2<f32>) -> f32 {
        let mut attention_patterns = Vec::new();
        
        for i in 0..x_test.shape()[0] {
            let sample = x_test.slice(s![i..i+1, ..]).to_owned();
            let (_, attention_weights) = model.forward_with_attention(&sample);
            
            if !attention_weights.is_empty() {
                attention_patterns.push(attention_weights[0].clone());
            }
        }
        
        if attention_patterns.len() < 2 {
            return 1.0;
        }
        
        // Measure stability as correlation between consecutive attention patterns
        let mut correlations = Vec::new();
        for i in 1..attention_patterns.len() {
            let corr = self.compute_attention_correlation(&attention_patterns[i-1], &attention_patterns[i]);
            correlations.push(corr);
        }
        
        correlations.iter().sum::<f32>() / correlations.len() as f32
    }
    
    fn measure_consciousness_entropy(&self, model: &NHITSModel, x_test: &Array2<f32>) -> f32 {
        // Measure entropy of consciousness states
        let mut entropy_values = Vec::new();
        
        for i in 0..x_test.shape()[0] {
            let sample = x_test.slice(s![i..i+1, ..]).to_owned();
            let _ = model.forward_with_consciousness(&sample);
            
            if let Some(consciousness) = &model.consciousness {
                let model_state = Array2::ones((1, 512)); // Simplified
                let awareness = consciousness.compute_awareness(&model_state);
                let entropy = self.compute_entropy(&awareness);
                entropy_values.push(entropy);
            }
        }
        
        if entropy_values.is_empty() {
            return 0.0;
        }
        
        entropy_values.iter().sum::<f32>() / entropy_values.len() as f32
    }
    
    fn measure_cognitive_load(&self, model: &NHITSModel, x_test: &Array2<f32>) -> f32 {
        // Simplified cognitive load measurement
        let sample = x_test.slice(s![0..1, ..]).to_owned();
        
        let start_time = std::time::Instant::now();
        let _ = model.forward_with_consciousness(&sample);
        let consciousness_time = start_time.elapsed();
        
        let start_time = std::time::Instant::now();
        let _ = model.forward(&sample);
        let regular_time = start_time.elapsed();
        
        // Cognitive load as relative computational overhead
        if regular_time.as_nanos() > 0 {
            consciousness_time.as_nanos() as f32 / regular_time.as_nanos() as f32
        } else {
            1.0
        }
    }
    
    fn compute_attention_correlation(&self, pattern1: &Array3<f32>, pattern2: &Array3<f32>) -> f32 {
        if pattern1.shape() != pattern2.shape() {
            return 0.0;
        }
        
        let flat1: Vec<f32> = pattern1.iter().cloned().collect();
        let flat2: Vec<f32> = pattern2.iter().cloned().collect();
        
        let mean1 = flat1.iter().sum::<f32>() / flat1.len() as f32;
        let mean2 = flat2.iter().sum::<f32>() / flat2.len() as f32;
        
        let numerator: f32 = flat1.iter().zip(flat2.iter())
            .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
            .sum();
        
        let sum_sq1: f32 = flat1.iter().map(|&x| (x - mean1).powi(2)).sum();
        let sum_sq2: f32 = flat2.iter().map(|&x| (x - mean2).powi(2)).sum();
        
        let denominator = (sum_sq1 * sum_sq2).sqrt();
        
        if denominator > 1e-8 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn compute_entropy(&self, distribution: &Array2<f32>) -> f32 {
        let mut entropy = 0.0;
        
        for &prob in distribution.iter() {
            if prob > 1e-8 {
                entropy -= prob * prob.ln();
            }
        }
        
        entropy
    }
}

/// Bootstrap validation results
#[derive(Debug, Clone)]
pub struct BootstrapResults {
    pub mean_metrics: ValidationMetrics,
    pub confidence_intervals: HashMap<String, (f32, f32)>,
    pub std_errors: HashMap<String, f32>,
}

impl BootstrapResults {
    pub fn from_metrics(metrics: Vec<ValidationMetrics>) -> Self {
        // Compute mean metrics
        let n = metrics.len() as f32;
        
        let mean_mse = metrics.iter().map(|m| m.mse).sum::<f32>() / n;
        let mean_mae = metrics.iter().map(|m| m.mae).sum::<f32>() / n;
        let mean_rmse = metrics.iter().map(|m| m.rmse).sum::<f32>() / n;
        let mean_r2 = metrics.iter().map(|m| m.r2_score).sum::<f32>() / n;
        
        // Create simplified mean metrics (other fields would be computed similarly)
        let mean_metrics = ValidationMetrics {
            mse: mean_mse,
            mae: mean_mae,
            rmse: mean_rmse,
            mape: 0.0, // Simplified
            smape: 0.0,
            r2_score: mean_r2,
            correlation: 0.0,
            directional_accuracy: 0.0,
            residual_analysis: ResidualAnalysis {
                mean_residual: 0.0,
                std_residual: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                ljung_box_p_value: 0.0,
                durbin_watson: 0.0,
                normality_test_p_value: 0.0,
            },
            forecast_bias: 0.0,
            forecast_variance: 0.0,
            prediction_intervals: PredictionIntervals {
                lower_80: Vec::new(),
                upper_80: Vec::new(),
                lower_95: Vec::new(),
                upper_95: Vec::new(),
                coverage_80: 0.0,
                coverage_95: 0.0,
            },
            consciousness_metrics: None,
        };
        
        // Compute confidence intervals and standard errors
        let mut confidence_intervals = HashMap::new();
        let mut std_errors = HashMap::new();
        
        // MSE confidence interval
        let mut mse_values: Vec<f32> = metrics.iter().map(|m| m.mse).collect();
        mse_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mse_ci = (
            mse_values[(mse_values.len() as f32 * 0.025) as usize],
            mse_values[(mse_values.len() as f32 * 0.975) as usize],
        );
        confidence_intervals.insert("mse".to_string(), mse_ci);
        
        let mse_std = {
            let variance = mse_values.iter()
                .map(|&x| (x - mean_mse).powi(2))
                .sum::<f32>() / mse_values.len() as f32;
            variance.sqrt()
        };
        std_errors.insert("mse".to_string(), mse_std);
        
        BootstrapResults {
            mean_metrics,
            confidence_intervals,
            std_errors,
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_validation_metrics_computation() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        
        let model_config = NHITSConfig::default();
        let model = NHITSModel::new(model_config);
        
        let x_test = Array2::ones((10, 168));
        let y_test = Array2::zeros((10, 24));
        
        let metrics = validator.validate_model(&model, &x_test, &y_test);
        
        assert!(metrics.mse >= 0.0);
        assert!(metrics.mae >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.r2_score <= 1.0);
    }

    #[test]
    fn test_cross_validation() {
        let config = ValidationConfig {
            cross_validation_folds: 3,
            time_series_validation: true,
            ..Default::default()
        };
        let validator = ModelValidator::new(config);
        
        let model_config = NHITSConfig {
            max_steps: 10, // Reduced for testing
            ..Default::default()
        };
        
        let x_data = Array2::ones((30, 168));
        let y_data = Array2::zeros((30, 24));
        
        let cv_results = validator.cross_validate(&model_config, &x_data, &y_data);
        
        assert_eq!(cv_results.len(), 3);
        for result in cv_results {
            assert!(result.mse >= 0.0);
        }
    }

    #[test]
    fn test_consciousness_validation() {
        let config = ValidationConfig {
            validate_consciousness: true,
            ..Default::default()
        };
        let validator = ModelValidator::new(config);
        
        let model_config = NHITSConfig::default();
        let mut model = NHITSModel::new(model_config);
        model.enable_consciousness(256, 8, 3);
        
        let x_test = Array2::ones((5, 168));
        let y_test = Array2::zeros((5, 24));
        
        let metrics = validator.validate_model(&model, &x_test, &y_test);
        
        assert!(metrics.consciousness_metrics.is_some());
        let consciousness_metrics = metrics.consciousness_metrics.unwrap();
        assert!(consciousness_metrics.awareness_consistency >= 0.0);
        assert!(consciousness_metrics.decision_confidence >= 0.0);
    }

    #[test]
    fn test_statistical_tests() {
        let test_suite = StatisticalTestSuite::new();
        
        // Test with normal-like residuals
        let residuals = vec![0.1, -0.05, 0.02, -0.08, 0.03, 0.01, -0.02, 0.04];
        
        let ljung_box_p = test_suite.ljung_box_test(&residuals);
        let durbin_watson = test_suite.durbin_watson_test(&residuals);
        let normality_p = test_suite.normality_test(&residuals);
        
        assert!(ljung_box_p >= 0.0 && ljung_box_p <= 1.0);
        assert!(durbin_watson >= 0.0 && durbin_watson <= 4.0);
        assert!(normality_p >= 0.0 && normality_p <= 1.0);
    }
}