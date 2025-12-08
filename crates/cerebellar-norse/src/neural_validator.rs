//! Neural Network Output Validation and Anomaly Detection
//! 
//! Provides comprehensive validation of neural network outputs to prevent
//! dangerous trading decisions from faulty or anomalous neural activity.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use nalgebra::{DVector, DMatrix};
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow, Context};
use tracing::{debug, info, warn, error};
use statrs::statistics::{Statistics, Distribution};
use statrs::distribution::{Normal, ChiSquared};

use crate::risk_management::{AnomalyType, RiskEvent};
use crate::{CerebellarCircuit, CircuitMetrics, LIFNeuron};

/// Neural network validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Output bounds (min, max)
    pub output_bounds: (f64, f64),
    /// Statistical anomaly detection threshold (z-score)
    pub anomaly_threshold: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum allowed rate of change
    pub max_rate_of_change: f64,
    /// Spike detection threshold (spikes per second)
    pub max_spike_rate: f64,
    /// Membrane potential bounds
    pub membrane_potential_bounds: (f64, f64),
    /// Connectivity anomaly threshold
    pub connectivity_threshold: f64,
    /// Model stability requirements
    pub stability_window_size: usize,
    /// Convergence detection settings
    pub convergence_tolerance: f64,
    /// Outlier detection sensitivity
    pub outlier_sensitivity: f64
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            output_bounds: (-10.0, 10.0),
            anomaly_threshold: 3.0,        // 3-sigma
            min_confidence: 0.7,           // 70% minimum confidence
            max_rate_of_change: 5.0,       // Maximum change per time step
            max_spike_rate: 1000.0,        // 1000 Hz maximum
            membrane_potential_bounds: (-2.0, 2.0),
            connectivity_threshold: 0.01,   // 1% connectivity change threshold
            stability_window_size: 100,     // Last 100 samples for stability
            convergence_tolerance: 0.001,   // Convergence tolerance
            outlier_sensitivity: 2.5       // Outlier detection threshold
        }
    }
}

/// Validation result with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation status
    pub is_valid: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Detected anomalies
    pub anomalies: Vec<DetectedAnomaly>,
    /// Statistical metrics
    pub statistics: ValidationStatistics,
    /// Recommendations for handling
    pub recommendations: Vec<String>,
    /// Validation timestamp
    pub timestamp: u64
}

/// Detected anomaly with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Severity score (0.0 to 1.0)
    pub severity: f64,
    /// Affected outputs/neurons
    pub affected_indices: Vec<usize>,
    /// Detailed description
    pub description: String,
    /// Statistical metrics
    pub metrics: AnomalyMetrics
}

/// Statistical metrics for anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyMetrics {
    /// Z-score for statistical anomalies
    pub z_score: Option<f64>,
    /// P-value for significance testing
    pub p_value: Option<f64>,
    /// Distance from expected range
    pub range_deviation: Option<f64>,
    /// Rate of change
    pub rate_of_change: Option<f64>
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    /// Output statistics
    pub output_stats: OutputStatistics,
    /// Neural activity statistics
    pub activity_stats: ActivityStatistics,
    /// Stability metrics
    pub stability_metrics: StabilityMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics
}

/// Output statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub skewness: f64,
    pub kurtosis: f64
}

/// Neural activity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityStatistics {
    pub spike_rate: f64,
    pub avg_membrane_potential: f64,
    pub active_neuron_percentage: f64,
    pub synchrony_index: f64,
    pub entropy: f64
}

/// Stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub variance_stability: f64,
    pub mean_stability: f64,
    pub convergence_rate: f64,
    pub oscillation_index: f64
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub processing_time_ns: u64,
    pub memory_usage_bytes: usize,
    pub validation_overhead_percent: f64
}

/// Historical data for trend analysis
#[derive(Debug)]
struct HistoricalData {
    outputs: VecDeque<(u64, Vec<f64>)>,
    membrane_potentials: VecDeque<(u64, Vec<f64>)>,
    spike_trains: VecDeque<(u64, Vec<bool>)>,
    statistics: VecDeque<(u64, OutputStatistics)>,
    max_history_size: usize
}

impl HistoricalData {
    fn new(max_size: usize) -> Self {
        Self {
            outputs: VecDeque::with_capacity(max_size),
            membrane_potentials: VecDeque::with_capacity(max_size),
            spike_trains: VecDeque::with_capacity(max_size),
            statistics: VecDeque::with_capacity(max_size),
            max_history_size: max_size
        }
    }

    fn add_sample(&mut self, timestamp: u64, outputs: Vec<f64>, 
                  membrane_potentials: Vec<f64>, spikes: Vec<bool>, stats: OutputStatistics) {
        // Add new samples
        self.outputs.push_back((timestamp, outputs));
        self.membrane_potentials.push_back((timestamp, membrane_potentials));
        self.spike_trains.push_back((timestamp, spikes));
        self.statistics.push_back((timestamp, stats));
        
        // Maintain size limits
        if self.outputs.len() > self.max_history_size {
            self.outputs.pop_front();
            self.membrane_potentials.pop_front();
            self.spike_trains.pop_front();
            self.statistics.pop_front();
        }
    }
}

/// Advanced neural network validator
pub struct NeuralValidator {
    /// Configuration
    config: ValidationConfig,
    /// Historical data for trend analysis
    history: Mutex<HistoricalData>,
    /// Expected output statistics (learned)
    expected_stats: RwLock<HashMap<String, OutputStatistics>>,
    /// Anomaly detection models
    anomaly_detectors: RwLock<HashMap<String, AnomalyDetector>>,
    /// Validation counters
    validation_counts: Mutex<ValidationCounts>
}

/// Validation statistics counters
#[derive(Debug, Default)]
struct ValidationCounts {
    total_validations: u64,
    anomalies_detected: u64,
    false_positives: u64,
    processing_time_total_ns: u64
}

/// Anomaly detection model
#[derive(Debug, Clone)]
struct AnomalyDetector {
    /// Moving average for baseline
    moving_average: f64,
    /// Moving variance for threshold
    moving_variance: f64,
    /// Sample count
    sample_count: u64,
    /// Learning rate
    learning_rate: f64
}

impl AnomalyDetector {
    fn new(learning_rate: f64) -> Self {
        Self {
            moving_average: 0.0,
            moving_variance: 1.0,
            sample_count: 0,
            learning_rate
        }
    }

    fn update(&mut self, value: f64) {
        self.sample_count += 1;
        let alpha = if self.sample_count < 100 {
            1.0 / self.sample_count as f64
        } else {
            self.learning_rate
        };

        let delta = value - self.moving_average;
        self.moving_average += alpha * delta;
        self.moving_variance = (1.0 - alpha) * self.moving_variance + alpha * delta * delta;
    }

    fn get_z_score(&self, value: f64) -> f64 {
        if self.moving_variance <= 0.0 {
            return 0.0;
        }
        (value - self.moving_average) / self.moving_variance.sqrt()
    }

    fn is_anomaly(&self, value: f64, threshold: f64) -> bool {
        self.get_z_score(value).abs() > threshold
    }
}

impl NeuralValidator {
    /// Create new neural validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            history: Mutex::new(HistoricalData::new(1000)),
            expected_stats: RwLock::new(HashMap::new()),
            anomaly_detectors: RwLock::new(HashMap::new()),
            validation_counts: Mutex::new(ValidationCounts::default())
        }
    }

    /// Comprehensive validation of neural network outputs
    pub fn validate_comprehensive(&self, 
                                 outputs: &[f64], 
                                 circuit_metrics: &CircuitMetrics,
                                 membrane_potentials: &[f64],
                                 spike_trains: &[bool]) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        // 1. Basic bounds checking
        let bounds_anomalies = self.check_output_bounds(outputs)?;
        
        // 2. Statistical anomaly detection
        let statistical_anomalies = self.detect_statistical_anomalies(outputs)?;
        
        // 3. Membrane potential validation
        let membrane_anomalies = self.validate_membrane_potentials(membrane_potentials)?;
        
        // 4. Spike pattern analysis
        let spike_anomalies = self.analyze_spike_patterns(spike_trains)?;
        
        // 5. Rate of change analysis
        let rate_anomalies = self.analyze_rate_of_change(outputs, timestamp)?;
        
        // 6. Neural activity coherence
        let coherence_anomalies = self.check_neural_coherence(circuit_metrics)?;
        
        // 7. Model convergence analysis
        let convergence_anomalies = self.check_model_convergence(outputs)?;
        
        // Combine all anomalies
        let mut all_anomalies = Vec::new();
        all_anomalies.extend(bounds_anomalies);
        all_anomalies.extend(statistical_anomalies);
        all_anomalies.extend(membrane_anomalies);
        all_anomalies.extend(spike_anomalies);
        all_anomalies.extend(rate_anomalies);
        all_anomalies.extend(coherence_anomalies);
        all_anomalies.extend(convergence_anomalies);
        
        // Calculate overall confidence
        let confidence = self.calculate_confidence(outputs, &all_anomalies)?;
        
        // Generate statistics
        let statistics = self.generate_statistics(outputs, membrane_potentials, 
                                                spike_trains, circuit_metrics)?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&all_anomalies, confidence);
        
        // Update historical data
        let output_stats = self.calculate_output_statistics(outputs)?;
        {
            let mut history = self.history.lock().unwrap();
            history.add_sample(timestamp, outputs.to_vec(), 
                             membrane_potentials.to_vec(), spike_trains.to_vec(), output_stats);
        }
        
        // Update learning models
        self.update_anomaly_detectors(outputs)?;
        
        // Record performance metrics
        let processing_time = start_time.elapsed().as_nanos() as u64;
        {
            let mut counts = self.validation_counts.lock().unwrap();
            counts.total_validations += 1;
            counts.anomalies_detected += all_anomalies.len() as u64;
            counts.processing_time_total_ns += processing_time;
        }
        
        let is_valid = all_anomalies.is_empty() && confidence >= self.config.min_confidence;
        
        Ok(ValidationResult {
            is_valid,
            confidence,
            anomalies: all_anomalies,
            statistics,
            recommendations,
            timestamp
        })
    }

    /// Check output bounds
    fn check_output_bounds(&self, outputs: &[f64]) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        
        for (i, &output) in outputs.iter().enumerate() {
            if !output.is_finite() {
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::OutputRangeAnomaly,
                    severity: 1.0,
                    affected_indices: vec![i],
                    description: format!("Output {} is not finite: {}", i, output),
                    metrics: AnomalyMetrics {
                        z_score: None,
                        p_value: None,
                        range_deviation: Some(f64::INFINITY),
                        rate_of_change: None
                    }
                });
            } else if output < self.config.output_bounds.0 || output > self.config.output_bounds.1 {
                let deviation = if output < self.config.output_bounds.0 {
                    self.config.output_bounds.0 - output
                } else {
                    output - self.config.output_bounds.1
                };
                
                let range_size = self.config.output_bounds.1 - self.config.output_bounds.0;
                let severity = (deviation / range_size).min(1.0);
                
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::OutputRangeAnomaly,
                    severity,
                    affected_indices: vec![i],
                    description: format!("Output {} ({}) outside bounds [{}, {}]", 
                                       i, output, self.config.output_bounds.0, self.config.output_bounds.1),
                    metrics: AnomalyMetrics {
                        z_score: None,
                        p_value: None,
                        range_deviation: Some(deviation),
                        rate_of_change: None
                    }
                });
            }
        }
        
        Ok(anomalies)
    }

    /// Detect statistical anomalies using learned patterns
    fn detect_statistical_anomalies(&self, outputs: &[f64]) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        let detectors = self.anomaly_detectors.read().unwrap();
        
        // Overall output statistics anomaly
        let mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let variance = outputs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / outputs.len() as f64;
        
        if let Some(mean_detector) = detectors.get("output_mean") {
            if mean_detector.is_anomaly(mean, self.config.anomaly_threshold) {
                let z_score = mean_detector.get_z_score(mean);
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::ActivityPatternAnomaly,
                    severity: (z_score.abs() / 5.0).min(1.0),
                    affected_indices: (0..outputs.len()).collect(),
                    description: format!("Output mean {} deviates significantly from baseline", mean),
                    metrics: AnomalyMetrics {
                        z_score: Some(z_score),
                        p_value: None,
                        range_deviation: None,
                        rate_of_change: None
                    }
                });
            }
        }
        
        if let Some(variance_detector) = detectors.get("output_variance") {
            if variance_detector.is_anomaly(variance, self.config.anomaly_threshold) {
                let z_score = variance_detector.get_z_score(variance);
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::ActivityPatternAnomaly,
                    severity: (z_score.abs() / 5.0).min(1.0),
                    affected_indices: (0..outputs.len()).collect(),
                    description: format!("Output variance {} deviates significantly from baseline", variance),
                    metrics: AnomalyMetrics {
                        z_score: Some(z_score),
                        p_value: None,
                        range_deviation: None,
                        rate_of_change: None
                    }
                });
            }
        }
        
        Ok(anomalies)
    }

    /// Validate membrane potentials
    fn validate_membrane_potentials(&self, membrane_potentials: &[f64]) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        
        for (i, &potential) in membrane_potentials.iter().enumerate() {
            if potential < self.config.membrane_potential_bounds.0 || 
               potential > self.config.membrane_potential_bounds.1 {
                
                let deviation = if potential < self.config.membrane_potential_bounds.0 {
                    self.config.membrane_potential_bounds.0 - potential
                } else {
                    potential - self.config.membrane_potential_bounds.1
                };
                
                let range_size = self.config.membrane_potential_bounds.1 - self.config.membrane_potential_bounds.0;
                let severity = (deviation / range_size).min(1.0);
                
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::MembranePotentialAnomaly,
                    severity,
                    affected_indices: vec![i],
                    description: format!("Membrane potential {} ({}) outside safe bounds", i, potential),
                    metrics: AnomalyMetrics {
                        z_score: None,
                        p_value: None,
                        range_deviation: Some(deviation),
                        rate_of_change: None
                    }
                });
            }
        }
        
        Ok(anomalies)
    }

    /// Analyze spike patterns for anomalies
    fn analyze_spike_patterns(&self, spike_trains: &[bool]) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        
        // Calculate spike rate
        let spike_count = spike_trains.iter().filter(|&&spike| spike).count();
        let spike_rate = spike_count as f64; // Spikes per time step
        
        if spike_rate > self.config.max_spike_rate {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::HyperActivityAnomaly,
                severity: ((spike_rate - self.config.max_spike_rate) / self.config.max_spike_rate).min(1.0),
                affected_indices: spike_trains.iter().enumerate()
                    .filter(|(_, &spike)| spike)
                    .map(|(i, _)| i)
                    .collect(),
                description: format!("Excessive spike activity: {} spikes", spike_count),
                metrics: AnomalyMetrics {
                    z_score: None,
                    p_value: None,
                    range_deviation: Some(spike_rate - self.config.max_spike_rate),
                    rate_of_change: None
                }
            });
        }
        
        // Check for abnormal spike patterns (e.g., all neurons spiking simultaneously)
        if spike_count == spike_trains.len() && spike_trains.len() > 10 {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::ActivityPatternAnomaly,
                severity: 0.8,
                affected_indices: (0..spike_trains.len()).collect(),
                description: "Abnormal synchronous spiking detected".to_string(),
                metrics: AnomalyMetrics {
                    z_score: None,
                    p_value: None,
                    range_deviation: None,
                    rate_of_change: None
                }
            });
        }
        
        Ok(anomalies)
    }

    /// Analyze rate of change in outputs
    fn analyze_rate_of_change(&self, outputs: &[f64], timestamp: u64) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        let history = self.history.lock().unwrap();
        
        if let Some((prev_timestamp, prev_outputs)) = history.outputs.back() {
            let time_diff = timestamp.saturating_sub(*prev_timestamp).max(1);
            
            for (i, (&current, &previous)) in outputs.iter().zip(prev_outputs.iter()).enumerate() {
                let rate = (current - previous).abs() / time_diff as f64;
                
                if rate > self.config.max_rate_of_change {
                    anomalies.push(DetectedAnomaly {
                        anomaly_type: AnomalyType::ActivityPatternAnomaly,
                        severity: (rate / self.config.max_rate_of_change).min(1.0),
                        affected_indices: vec![i],
                        description: format!("Rapid change in output {}: rate {}", i, rate),
                        metrics: AnomalyMetrics {
                            z_score: None,
                            p_value: None,
                            range_deviation: None,
                            rate_of_change: Some(rate)
                        }
                    });
                }
            }
        }
        
        Ok(anomalies)
    }

    /// Check neural network coherence
    fn check_neural_coherence(&self, circuit_metrics: &CircuitMetrics) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        
        // Check for unrealistic activity levels
        let total_neurons = circuit_metrics.total_neurons;
        let total_active = circuit_metrics.granule_stats.active_neurons +
                          circuit_metrics.purkinje_stats.active_neurons +
                          circuit_metrics.golgi_stats.active_neurons +
                          circuit_metrics.dcn_stats.active_neurons;
        
        let activity_ratio = total_active as f64 / total_neurons as f64;
        
        // Check for too high activity (>80% of neurons active)
        if activity_ratio > 0.8 {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::HyperActivityAnomaly,
                severity: ((activity_ratio - 0.8) / 0.2).min(1.0),
                affected_indices: Vec::new(),
                description: format!("Excessive neural activity: {:.1}% of neurons active", activity_ratio * 100.0),
                metrics: AnomalyMetrics {
                    z_score: None,
                    p_value: None,
                    range_deviation: Some(activity_ratio - 0.8),
                    rate_of_change: None
                }
            });
        }
        
        // Check for too low activity (<1% of neurons active)
        if activity_ratio < 0.01 {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::ActivityPatternAnomaly,
                severity: (0.01 - activity_ratio) / 0.01,
                affected_indices: Vec::new(),
                description: format!("Insufficient neural activity: {:.1}% of neurons active", activity_ratio * 100.0),
                metrics: AnomalyMetrics {
                    z_score: None,
                    p_value: None,
                    range_deviation: Some(0.01 - activity_ratio),
                    rate_of_change: None
                }
            });
        }
        
        Ok(anomalies)
    }

    /// Check model convergence
    fn check_model_convergence(&self, outputs: &[f64]) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();
        let history = self.history.lock().unwrap();
        
        if history.statistics.len() >= self.config.stability_window_size {
            let recent_stats: Vec<&OutputStatistics> = history.statistics.iter()
                .rev()
                .take(self.config.stability_window_size)
                .map(|(_, stats)| stats)
                .collect();
            
            // Check variance stability
            let variances: Vec<f64> = recent_stats.iter()
                .map(|stats| stats.std_dev.powi(2))
                .collect();
            
            let variance_of_variances = {
                let mean_var = variances.iter().sum::<f64>() / variances.len() as f64;
                variances.iter()
                    .map(|&var| (var - mean_var).powi(2))
                    .sum::<f64>() / variances.len() as f64
            };
            
            if variance_of_variances.sqrt() > self.config.convergence_tolerance {
                anomalies.push(DetectedAnomaly {
                    anomaly_type: AnomalyType::ConvergenceFailure,
                    severity: (variance_of_variances.sqrt() / self.config.convergence_tolerance).min(1.0),
                    affected_indices: (0..outputs.len()).collect(),
                    description: "Model outputs showing poor convergence".to_string(),
                    metrics: AnomalyMetrics {
                        z_score: None,
                        p_value: None,
                        range_deviation: Some(variance_of_variances.sqrt() - self.config.convergence_tolerance),
                        rate_of_change: None
                    }
                });
            }
        }
        
        Ok(anomalies)
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, outputs: &[f64], anomalies: &[DetectedAnomaly]) -> Result<f64> {
        let mut confidence = 1.0;
        
        // Reduce confidence based on anomalies
        for anomaly in anomalies {
            confidence *= 1.0 - (anomaly.severity * 0.3); // Max 30% reduction per anomaly
        }
        
        // Additional confidence factors
        let output_stats = self.calculate_output_statistics(outputs)?;
        
        // Penalize high variance
        let variance_penalty = (output_stats.std_dev / 5.0).min(0.2);
        confidence *= 1.0 - variance_penalty;
        
        // Penalize extreme values
        let extreme_penalty = if output_stats.max > 8.0 || output_stats.min < -8.0 { 0.1 } else { 0.0 };
        confidence *= 1.0 - extreme_penalty;
        
        Ok(confidence.max(0.0))
    }

    /// Generate comprehensive statistics
    fn generate_statistics(&self, outputs: &[f64], membrane_potentials: &[f64], 
                          spike_trains: &[bool], circuit_metrics: &CircuitMetrics) -> Result<ValidationStatistics> {
        let output_stats = self.calculate_output_statistics(outputs)?;
        
        let activity_stats = ActivityStatistics {
            spike_rate: spike_trains.iter().filter(|&&s| s).count() as f64,
            avg_membrane_potential: membrane_potentials.iter().sum::<f64>() / membrane_potentials.len() as f64,
            active_neuron_percentage: {
                let total_active = circuit_metrics.granule_stats.active_neurons +
                                 circuit_metrics.purkinje_stats.active_neurons +
                                 circuit_metrics.golgi_stats.active_neurons +
                                 circuit_metrics.dcn_stats.active_neurons;
                (total_active as f64 / circuit_metrics.total_neurons as f64) * 100.0
            },
            synchrony_index: self.calculate_synchrony_index(spike_trains)?,
            entropy: self.calculate_entropy(outputs)?
        };
        
        let stability_metrics = self.calculate_stability_metrics()?;
        
        let performance_metrics = PerformanceMetrics {
            processing_time_ns: 0, // Would be set by caller
            memory_usage_bytes: std::mem::size_of_val(outputs) + 
                               std::mem::size_of_val(membrane_potentials) +
                               std::mem::size_of_val(spike_trains),
            validation_overhead_percent: 5.0 // Typical overhead
        };
        
        Ok(ValidationStatistics {
            output_stats,
            activity_stats,
            stability_metrics,
            performance_metrics
        })
    }

    /// Calculate output statistics
    fn calculate_output_statistics(&self, outputs: &[f64]) -> Result<OutputStatistics> {
        if outputs.is_empty() {
            return Ok(OutputStatistics {
                mean: 0.0, std_dev: 0.0, min: 0.0, max: 0.0,
                median: 0.0, skewness: 0.0, kurtosis: 0.0
            });
        }
        
        let mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let variance = outputs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / outputs.len() as f64;
        let std_dev = variance.sqrt();
        
        let min = outputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let mut sorted = outputs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        
        // Calculate skewness and kurtosis
        let n = outputs.len() as f64;
        let moment3 = outputs.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;
        let moment4 = outputs.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n;
        
        let skewness = moment3;
        let kurtosis = moment4 - 3.0; // Excess kurtosis
        
        Ok(OutputStatistics {
            mean, std_dev, min, max, median, skewness, kurtosis
        })
    }

    /// Calculate synchrony index for spike trains
    fn calculate_synchrony_index(&self, spike_trains: &[bool]) -> Result<f64> {
        if spike_trains.is_empty() {
            return Ok(0.0);
        }
        
        let spike_count = spike_trains.iter().filter(|&&s| s).count();
        let max_synchrony = spike_trains.len();
        
        Ok(spike_count as f64 / max_synchrony as f64)
    }

    /// Calculate entropy of outputs
    fn calculate_entropy(&self, outputs: &[f64]) -> Result<f64> {
        if outputs.is_empty() {
            return Ok(0.0);
        }
        
        // Discretize outputs into bins for entropy calculation
        let bins = 10;
        let min_val = outputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        if (max_val - min_val).abs() < 1e-10 {
            return Ok(0.0); // All values the same
        }
        
        let bin_size = (max_val - min_val) / bins as f64;
        let mut bin_counts = vec![0; bins];
        
        for &output in outputs {
            let bin_index = ((output - min_val) / bin_size).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            bin_counts[bin_index] += 1;
        }
        
        let total = outputs.len() as f64;
        let entropy = bin_counts.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum();
        
        Ok(entropy)
    }

    /// Calculate stability metrics
    fn calculate_stability_metrics(&self) -> Result<StabilityMetrics> {
        let history = self.history.lock().unwrap();
        
        if history.statistics.len() < 2 {
            return Ok(StabilityMetrics {
                variance_stability: 1.0,
                mean_stability: 1.0,
                convergence_rate: 0.0,
                oscillation_index: 0.0
            });
        }
        
        let recent_stats: Vec<&OutputStatistics> = history.statistics.iter()
            .rev()
            .take(self.config.stability_window_size)
            .map(|(_, stats)| stats)
            .collect();
        
        // Variance stability
        let variances: Vec<f64> = recent_stats.iter().map(|stats| stats.std_dev.powi(2)).collect();
        let variance_stability = if variances.len() > 1 {
            let mean_var = variances.iter().sum::<f64>() / variances.len() as f64;
            let var_of_var = variances.iter()
                .map(|&var| (var - mean_var).powi(2))
                .sum::<f64>() / variances.len() as f64;
            1.0 / (1.0 + var_of_var.sqrt())
        } else {
            1.0
        };
        
        // Mean stability
        let means: Vec<f64> = recent_stats.iter().map(|stats| stats.mean).collect();
        let mean_stability = if means.len() > 1 {
            let mean_of_means = means.iter().sum::<f64>() / means.len() as f64;
            let var_of_means = means.iter()
                .map(|&mean| (mean - mean_of_means).powi(2))
                .sum::<f64>() / means.len() as f64;
            1.0 / (1.0 + var_of_means.sqrt())
        } else {
            1.0
        };
        
        Ok(StabilityMetrics {
            variance_stability,
            mean_stability,
            convergence_rate: 0.9, // Placeholder
            oscillation_index: 0.1  // Placeholder
        })
    }

    /// Generate recommendations based on anomalies
    fn generate_recommendations(&self, anomalies: &[DetectedAnomaly], confidence: f64) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if confidence < 0.5 {
            recommendations.push("CRITICAL: Neural network confidence extremely low. Halt trading immediately.".to_string());
        } else if confidence < 0.7 {
            recommendations.push("WARNING: Neural network confidence below threshold. Consider reducing position sizes.".to_string());
        }
        
        for anomaly in anomalies {
            match anomaly.anomaly_type {
                AnomalyType::OutputRangeAnomaly => {
                    if anomaly.severity > 0.8 {
                        recommendations.push("CRITICAL: Output values severely out of bounds. Emergency shutdown recommended.".to_string());
                    } else {
                        recommendations.push("WARNING: Output values outside expected range. Monitor closely.".to_string());
                    }
                },
                AnomalyType::HyperActivityAnomaly => {
                    recommendations.push("WARNING: Excessive neural activity detected. Check for feedback loops.".to_string());
                },
                AnomalyType::ConvergenceFailure => {
                    recommendations.push("WARNING: Model showing poor convergence. Consider retraining.".to_string());
                },
                AnomalyType::MembranePotentialAnomaly => {
                    recommendations.push("WARNING: Unstable membrane potentials. Check neuron parameters.".to_string());
                },
                _ => {
                    recommendations.push(format!("Monitor anomaly: {}", anomaly.description));
                }
            }
        }
        
        if recommendations.is_empty() {
            recommendations.push("Neural network outputs appear normal. Continue monitoring.".to_string());
        }
        
        recommendations
    }

    /// Update anomaly detection models with new data
    fn update_anomaly_detectors(&self, outputs: &[f64]) -> Result<()> {
        let mut detectors = self.anomaly_detectors.write().unwrap();
        
        // Update or create output mean detector
        let mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        detectors.entry("output_mean".to_string())
            .or_insert_with(|| AnomalyDetector::new(0.01))
            .update(mean);
        
        // Update or create output variance detector
        let variance = outputs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / outputs.len() as f64;
        detectors.entry("output_variance".to_string())
            .or_insert_with(|| AnomalyDetector::new(0.01))
            .update(variance);
        
        Ok(())
    }

    /// Get validation performance metrics
    pub fn get_performance_metrics(&self) -> ValidationCounts {
        self.validation_counts.lock().unwrap().clone()
    }

    /// Reset learning state (for testing or reinitialization)
    pub fn reset_learning(&self) -> Result<()> {
        {
            let mut history = self.history.lock().unwrap();
            history.outputs.clear();
            history.membrane_potentials.clear();
            history.spike_trains.clear();
            history.statistics.clear();
        }
        
        {
            let mut detectors = self.anomaly_detectors.write().unwrap();
            detectors.clear();
        }
        
        {
            let mut expected = self.expected_stats.write().unwrap();
            expected.clear();
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = NeuralValidator::new(config);
        
        let metrics = validator.get_performance_metrics();
        assert_eq!(metrics.total_validations, 0);
    }

    #[test]
    fn test_output_statistics() {
        let config = ValidationConfig::default();
        let validator = NeuralValidator::new(config);
        
        let outputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = validator.calculate_output_statistics(&outputs).unwrap();
        
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0);
    }

    #[test]
    fn test_bounds_checking() {
        let config = ValidationConfig::default();
        let validator = NeuralValidator::new(config);
        
        // Valid outputs
        let valid_outputs = vec![1.0, 2.0, 3.0];
        let anomalies = validator.check_output_bounds(&valid_outputs).unwrap();
        assert!(anomalies.is_empty());
        
        // Invalid outputs
        let invalid_outputs = vec![15.0, 2.0, 3.0]; // 15.0 > 10.0 (upper bound)
        let anomalies = validator.check_output_bounds(&invalid_outputs).unwrap();
        assert_eq!(anomalies.len(), 1);
        assert!(matches!(anomalies[0].anomaly_type, AnomalyType::OutputRangeAnomaly));
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(0.1);
        
        // Train with normal values
        for _ in 0..100 {
            detector.update(5.0);
        }
        
        // Test anomaly detection
        assert!(!detector.is_anomaly(5.1, 3.0)); // Normal value
        assert!(detector.is_anomaly(10.0, 3.0));  // Anomalous value
    }
}