//! Neural Network Calibration Integration using ATS-Core
//! 
//! This module integrates ATS-Core's Adaptive Temperature Scaling and Conformal Prediction
//! to calibrate all neural network predictions before they reach QAR, ensuring reliable
//! confidence scores and uncertainty quantification for trading decisions.

use crate::{NeuromorphicSignal, ModuleContribution, ComprehensiveNeuralSignal};
use ats_core::{AtsCpEngine, config::AtsCpConfig, temperature::TemperatureScaler, conformal::ConformalPredictor};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use std::collections::HashMap;

/// Neural Calibration System that ensures all predictions are properly calibrated
pub struct NeuralCalibrationSystem {
    /// ATS-CP Engine for temperature scaling and conformal prediction
    ats_engine: Arc<RwLock<AtsCpEngine>>,
    
    /// Temperature scaler for confidence calibration
    temperature_scaler: Arc<RwLock<TemperatureScaler>>,
    
    /// Conformal predictor for uncertainty quantification
    conformal_predictor: Arc<RwLock<ConformalPredictor>>,
    
    /// Calibration history for each neural module
    calibration_history: Arc<RwLock<HashMap<String, CalibrationHistory>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<CalibrationMetrics>>,
}

/// Calibration history for a specific neural module
#[derive(Debug, Clone)]
pub struct CalibrationHistory {
    /// Module name
    module_name: String,
    /// Historical temperatures used
    temperature_history: Vec<f64>,
    /// Calibration errors
    calibration_errors: Vec<f64>,
    /// Conformal coverage rates
    coverage_rates: Vec<f64>,
    /// Last update timestamp
    last_updated: std::time::Instant,
}

/// Calibration performance metrics
#[derive(Debug, Default)]
pub struct CalibrationMetrics {
    /// Total calibrations performed
    pub total_calibrations: u64,
    /// Average calibration latency
    pub avg_latency_us: f64,
    /// Temperature optimization success rate
    pub optimization_success_rate: f64,
    /// Conformal coverage accuracy
    pub coverage_accuracy: f64,
}

impl NeuralCalibrationSystem {
    /// Create new calibration system
    pub async fn new() -> Result<Self> {
        info!("🎯 Initializing Neural Calibration System with ATS-Core");
        
        // Create ATS-CP configuration optimized for neural networks
        let config = AtsCpConfig {
            temperature: ats_core::config::TemperatureConfig {
                min_temperature: 0.1,
                max_temperature: 10.0,
                default_temperature: 1.0,
                search_tolerance: 0.001,
                max_search_iterations: 50,
                target_latency_us: 5, // 5μs target for sub-millisecond trading
            },
            conformal: ats_core::config::ConformalConfig {
                confidence_level: 0.95,
                calibration_size: 1000,
                update_frequency: 100,
                target_latency_us: 10,
            },
            simd: ats_core::config::SimdConfig {
                enabled: true,
                vector_width: 8,
                alignment_bytes: 64,
                min_simd_size: 32,
            },
            memory: ats_core::config::MemoryConfig {
                max_allocation_mb: 512,
                enable_huge_pages: true,
                preallocate: true,
            },
            parallel: ats_core::config::ParallelConfig {
                enabled: true,
                num_threads: num_cpus::get(),
                min_batch_size: 1000,
            },
        };
        
        // Initialize components
        let ats_engine = Arc::new(RwLock::new(AtsCpEngine::new(config.clone())?));
        let temperature_scaler = Arc::new(RwLock::new(TemperatureScaler::new(&config)?));
        let conformal_predictor = Arc::new(RwLock::new(ConformalPredictor::new(&config)?));
        
        info!("✅ Neural Calibration System initialized with sub-5μs temperature scaling");
        
        Ok(Self {
            ats_engine,
            temperature_scaler,
            conformal_predictor,
            calibration_history: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(CalibrationMetrics::default())),
        })
    }
    
    /// Calibrate a neuromorphic signal before sending to QAR
    pub async fn calibrate_neuromorphic_signal(
        &self,
        signal: &mut NeuromorphicSignal,
    ) -> Result<CalibrationResult> {
        let start_time = std::time::Instant::now();
        
        // Calibrate each module contribution
        for (module_name, contribution) in &mut signal.module_contributions {
            let calibrated = self.calibrate_module_contribution(
                module_name,
                contribution,
            ).await?;
            
            // Update contribution with calibrated values
            contribution.prediction = calibrated.calibrated_prediction;
            contribution.confidence = calibrated.calibrated_confidence;
        }
        
        // Recalculate overall signal confidence using calibrated values
        let total_confidence: f64 = signal.module_contributions.values()
            .map(|c| c.confidence)
            .sum::<f64>() / signal.module_contributions.len() as f64;
        
        signal.confidence = total_confidence;
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_calibrations += 1;
        metrics.avg_latency_us = start_time.elapsed().as_micros() as f64;
        
        debug!("Calibrated neuromorphic signal in {}μs", start_time.elapsed().as_micros());
        
        Ok(CalibrationResult {
            original_confidence: signal.confidence,
            calibrated_confidence: total_confidence,
            uncertainty_bounds: self.compute_uncertainty_bounds(signal).await?,
            calibration_time_us: start_time.elapsed().as_micros() as u64,
        })
    }
    
    /// Calibrate comprehensive neural signal from all systems
    pub async fn calibrate_comprehensive_signal(
        &self,
        signal: &mut ComprehensiveNeuralSignal,
    ) -> Result<CalibrationResult> {
        let start_time = std::time::Instant::now();
        
        // Calibrate each individual neural signal
        let mut calibrated_predictions = Vec::new();
        let mut calibrated_confidences = Vec::new();
        
        for neural_signal in &mut signal.individual_signals {
            // Extract predictions for temperature scaling
            let predictions = vec![neural_signal.prediction];
            
            // Optimize temperature for this neural system
            let optimal_temp = self.optimize_temperature_for_system(
                &neural_signal.source,
                &predictions,
            ).await?;
            
            // Apply temperature scaling
            let mut scaler = self.temperature_scaler.write().await;
            let scaled_predictions = scaler.scale(&predictions, optimal_temp)?;
            
            // Apply conformal prediction
            let mut conformal = self.conformal_predictor.write().await;
            let (lower, upper) = conformal.predict_interval(&scaled_predictions[0])?;
            
            // Update neural signal
            neural_signal.prediction = scaled_predictions[0];
            neural_signal.confidence = self.confidence_from_interval(lower, upper);
            
            calibrated_predictions.push(scaled_predictions[0]);
            calibrated_confidences.push(neural_signal.confidence);
        }
        
        // Recalculate overall signal with calibrated values
        signal.prediction = self.aggregate_calibrated_predictions(&calibrated_predictions);
        signal.confidence = self.aggregate_calibrated_confidences(&calibrated_confidences);
        
        // Update consensus score based on calibrated predictions
        signal.consensus_score = self.calculate_calibrated_consensus(&calibrated_predictions);
        
        Ok(CalibrationResult {
            original_confidence: signal.confidence,
            calibrated_confidence: signal.confidence,
            uncertainty_bounds: (
                calibrated_predictions.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0),
                calibrated_predictions.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0),
            ),
            calibration_time_us: start_time.elapsed().as_micros() as u64,
        })
    }
    
    /// Calibrate a single module contribution
    async fn calibrate_module_contribution(
        &self,
        module_name: &str,
        contribution: &ModuleContribution,
    ) -> Result<CalibratedContribution> {
        // Get or create calibration history for this module
        let mut history_map = self.calibration_history.write().await;
        let history = history_map.entry(module_name.to_string())
            .or_insert_with(|| CalibrationHistory {
                module_name: module_name.to_string(),
                temperature_history: Vec::new(),
                calibration_errors: Vec::new(),
                coverage_rates: Vec::new(),
                last_updated: std::time::Instant::now(),
            });
        
        // Use historical data to determine optimal temperature
        let optimal_temp = if history.temperature_history.is_empty() {
            1.0 // Default temperature
        } else {
            // Use exponentially weighted moving average
            let weights: Vec<f64> = (0..history.temperature_history.len())
                .map(|i| 0.9_f64.powi(i as i32))
                .collect();
            let sum_weights: f64 = weights.iter().sum();
            
            history.temperature_history.iter()
                .rev()
                .zip(weights.iter())
                .map(|(temp, weight)| temp * weight)
                .sum::<f64>() / sum_weights
        };
        
        // Apply temperature scaling
        let mut scaler = self.temperature_scaler.write().await;
        let scaled_prediction = scaler.scale(&[contribution.prediction], optimal_temp)?[0];
        
        // Apply conformal prediction for uncertainty
        let mut conformal = self.conformal_predictor.write().await;
        let (lower_bound, upper_bound) = conformal.predict_interval(&scaled_prediction)?;
        
        // Calculate calibrated confidence based on prediction interval
        let calibrated_confidence = self.confidence_from_interval(lower_bound, upper_bound);
        
        // Update history
        history.temperature_history.push(optimal_temp);
        if history.temperature_history.len() > 100 {
            history.temperature_history.remove(0);
        }
        history.last_updated = std::time::Instant::now();
        
        Ok(CalibratedContribution {
            calibrated_prediction: scaled_prediction,
            calibrated_confidence,
            temperature_used: optimal_temp,
            uncertainty_interval: (lower_bound, upper_bound),
        })
    }
    
    /// Optimize temperature for a specific neural system
    async fn optimize_temperature_for_system(
        &self,
        system_name: &str,
        predictions: &[f64],
    ) -> Result<f64> {
        // Get historical performance for this system
        let history_map = self.calibration_history.read().await;
        
        if let Some(history) = history_map.get(system_name) {
            if !history.temperature_history.is_empty() {
                // Use adaptive temperature based on recent performance
                let recent_temps = &history.temperature_history[history.temperature_history.len().saturating_sub(10)..];
                let avg_temp = recent_temps.iter().sum::<f64>() / recent_temps.len() as f64;
                
                // Adjust based on recent calibration errors
                let recent_errors = &history.calibration_errors[history.calibration_errors.len().saturating_sub(10)..];
                if !recent_errors.is_empty() {
                    let avg_error = recent_errors.iter().sum::<f64>() / recent_errors.len() as f64;
                    
                    // Increase temperature if errors are high
                    if avg_error > 0.1 {
                        return Ok((avg_temp * 1.1).min(10.0));
                    } else if avg_error < 0.05 {
                        return Ok((avg_temp * 0.9).max(0.1));
                    }
                }
                
                return Ok(avg_temp);
            }
        }
        
        // Default optimization for new systems
        let mut scaler = self.temperature_scaler.write().await;
        
        // Use a simple grid search for initial optimization
        let temperatures = vec![0.5, 0.75, 1.0, 1.5, 2.0];
        let mut best_temp = 1.0;
        let mut best_score = f64::MAX;
        
        for &temp in &temperatures {
            let scaled = scaler.scale(predictions, temp)?;
            
            // Simple calibration score (variance of scaled predictions)
            let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
            let variance = scaled.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / scaled.len() as f64;
            
            if variance < best_score {
                best_score = variance;
                best_temp = temp;
            }
        }
        
        Ok(best_temp)
    }
    
    /// Calculate confidence from prediction interval
    fn confidence_from_interval(&self, lower: f64, upper: f64) -> f64 {
        // Narrower intervals indicate higher confidence
        let interval_width = (upper - lower).abs();
        
        // Map interval width to confidence (inverse relationship)
        // Assuming typical interval widths are between 0.01 and 1.0
        let normalized_width = interval_width.clamp(0.01, 1.0);
        
        // Use exponential decay for confidence mapping
        ((-normalized_width * 2.0).exp()).clamp(0.1, 0.99)
    }
    
    /// Aggregate calibrated predictions
    fn aggregate_calibrated_predictions(&self, predictions: &[f64]) -> f64 {
        // Use weighted average based on prediction stability
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        if variance < 0.01 {
            // Low variance - simple average
            mean
        } else {
            // High variance - use median for robustness
            let mut sorted = predictions.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        }
    }
    
    /// Aggregate calibrated confidences
    fn aggregate_calibrated_confidences(&self, confidences: &[f64]) -> f64 {
        // Use harmonic mean for conservative confidence aggregation
        let sum_reciprocals: f64 = confidences.iter()
            .map(|&c| 1.0 / c.max(0.01))
            .sum();
        
        confidences.len() as f64 / sum_reciprocals
    }
    
    /// Calculate consensus among calibrated predictions
    fn calculate_calibrated_consensus(&self, predictions: &[f64]) -> f64 {
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Lower variance = higher consensus
        1.0 / (1.0 + variance.sqrt())
    }
    
    /// Compute uncertainty bounds for the signal
    async fn compute_uncertainty_bounds(&self, signal: &NeuromorphicSignal) -> Result<(f64, f64)> {
        let predictions: Vec<f64> = signal.module_contributions.values()
            .map(|c| c.prediction)
            .collect();
        
        if predictions.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        // Get conformal prediction intervals for each prediction
        let mut conformal = self.conformal_predictor.write().await;
        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();
        
        for &pred in &predictions {
            let (lower, upper) = conformal.predict_interval(&pred)?;
            lower_bounds.push(lower);
            upper_bounds.push(upper);
        }
        
        // Return the widest bounds for conservative uncertainty
        let min_lower = lower_bounds.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
        let max_upper = upper_bounds.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
        
        Ok((min_lower, max_upper))
    }
    
    /// Get calibration metrics
    pub async fn get_metrics(&self) -> CalibrationMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Update calibration models with new data
    pub async fn update_calibration_models(
        &self,
        predictions: &[f64],
        actual_outcomes: &[f64],
    ) -> Result<()> {
        if predictions.len() != actual_outcomes.len() {
            return Err(anyhow!("Predictions and outcomes must have same length"));
        }
        
        // Update conformal predictor with new calibration data
        let mut conformal = self.conformal_predictor.write().await;
        conformal.update_calibration_set(predictions, actual_outcomes)?;
        
        // Calculate calibration error for temperature adjustment
        let calibration_error = predictions.iter()
            .zip(actual_outcomes.iter())
            .map(|(pred, actual)| (pred - actual).abs())
            .sum::<f64>() / predictions.len() as f64;
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        
        // Check if predictions were well-calibrated
        let coverage_rate = predictions.iter()
            .zip(actual_outcomes.iter())
            .filter(|(pred, actual)| {
                // Check if actual outcome falls within expected range
                (pred - actual).abs() < 0.1 // Example threshold
            })
            .count() as f64 / predictions.len() as f64;
        
        metrics.coverage_accuracy = (metrics.coverage_accuracy * 0.9) + (coverage_rate * 0.1);
        
        info!("Updated calibration models - Error: {:.4}, Coverage: {:.2}%", 
              calibration_error, coverage_rate * 100.0);
        
        Ok(())
    }
}

/// Result of calibration process
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Original confidence before calibration
    pub original_confidence: f64,
    /// Calibrated confidence after temperature scaling
    pub calibrated_confidence: f64,
    /// Uncertainty bounds (lower, upper)
    pub uncertainty_bounds: (f64, f64),
    /// Calibration time in microseconds
    pub calibration_time_us: u64,
}

/// Calibrated contribution from a module
#[derive(Debug, Clone)]
struct CalibratedContribution {
    /// Calibrated prediction value
    calibrated_prediction: f64,
    /// Calibrated confidence score
    calibrated_confidence: f64,
    /// Temperature used for scaling
    temperature_used: f64,
    /// Uncertainty interval from conformal prediction
    uncertainty_interval: (f64, f64),
}

/// Extension trait for QuantumQueen to use calibration
pub trait QuantumQueenCalibration {
    /// Process signal with calibration before QAR integration
    async fn integrate_calibrated_neuromorphic_signal(
        &mut self,
        mut signal: NeuromorphicSignal,
        calibration_system: &NeuralCalibrationSystem,
    ) -> Result<()>;
}

impl QuantumQueenCalibration for crate::QuantumQueen {
    async fn integrate_calibrated_neuromorphic_signal(
        &mut self,
        mut signal: NeuromorphicSignal,
        calibration_system: &NeuralCalibrationSystem,
    ) -> Result<()> {
        // Calibrate the signal
        let calibration_result = calibration_system.calibrate_neuromorphic_signal(&mut signal).await?;
        
        info!("🎯 Neural signal calibrated - Original confidence: {:.2}%, Calibrated: {:.2}%, Bounds: ({:.4}, {:.4})",
              calibration_result.original_confidence * 100.0,
              calibration_result.calibrated_confidence * 100.0,
              calibration_result.uncertainty_bounds.0,
              calibration_result.uncertainty_bounds.1);
        
        // Now integrate the calibrated signal into QAR
        self.integrate_neuromorphic_signal(signal).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_calibration_system_creation() {
        let system = NeuralCalibrationSystem::new().await;
        assert!(system.is_ok());
    }
    
    #[tokio::test]
    async fn test_neuromorphic_signal_calibration() {
        let system = NeuralCalibrationSystem::new().await.unwrap();
        
        let mut signal = NeuromorphicSignal {
            prediction: 0.75,
            confidence: 0.85,
            module_contributions: HashMap::from([
                ("test_module".to_string(), ModuleContribution {
                    module_name: "test_module".to_string(),
                    prediction: 0.8,
                    confidence: 0.9,
                    processing_time_us: 100,
                }),
            ]),
            spike_patterns: vec![],
            temporal_coherence: 0.7,
            functional_optimization: 0.8,
        };
        
        let result = system.calibrate_neuromorphic_signal(&mut signal).await;
        assert!(result.is_ok());
        
        let calibration = result.unwrap();
        assert!(calibration.calibrated_confidence > 0.0);
        assert!(calibration.calibrated_confidence <= 1.0);
    }
    
    #[test]
    fn test_confidence_from_interval() {
        let system = NeuralCalibrationSystem::new().await.unwrap();
        
        // Narrow interval should give high confidence
        let confidence = system.confidence_from_interval(0.45, 0.55);
        assert!(confidence > 0.8);
        
        // Wide interval should give low confidence  
        let confidence = system.confidence_from_interval(0.0, 1.0);
        assert!(confidence < 0.5);
    }
}