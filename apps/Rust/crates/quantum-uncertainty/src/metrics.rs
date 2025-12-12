//! # Quantum Uncertainty Metrics
//!
//! This module implements comprehensive metrics for tracking and evaluating
//! quantum uncertainty quantification performance.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{UncertaintyEstimate, ConformalPredictionIntervals, OptimizedMeasurements};

/// Comprehensive quantum metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Uncertainty quantification metrics
    pub uncertainty_metrics: UncertaintyMetrics,
    /// Conformal prediction metrics
    pub conformal_metrics: ConformalMetrics,
    /// Measurement optimization metrics
    pub measurement_metrics: MeasurementMetrics,
    /// Quantum advantage metrics
    pub quantum_advantage: f64,
    /// Computational performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Circuit fidelity metrics
    pub fidelity_metrics: FidelityMetrics,
    /// Feature extraction metrics
    pub feature_metrics: FeatureMetrics,
    /// Correlation analysis metrics
    pub correlation_metrics: CorrelationMetrics,
    /// System health metrics
    pub system_metrics: SystemMetrics,
    /// Metrics metadata
    pub metadata: MetricsMetadata,
}

impl QuantumMetrics {
    /// Create new quantum metrics
    pub fn new() -> Self {
        Self {
            uncertainty_metrics: UncertaintyMetrics::new(),
            conformal_metrics: ConformalMetrics::new(),
            measurement_metrics: MeasurementMetrics::new(),
            quantum_advantage: 0.0,
            performance_metrics: PerformanceMetrics::new(),
            fidelity_metrics: FidelityMetrics::new(),
            feature_metrics: FeatureMetrics::new(),
            correlation_metrics: CorrelationMetrics::new(),
            system_metrics: SystemMetrics::new(),
            metadata: MetricsMetadata::new(),
        }
    }

    /// Update uncertainty metrics
    pub fn update_uncertainty_metrics(&mut self, estimates: &[UncertaintyEstimate]) {
        self.uncertainty_metrics.update(estimates);
        self.metadata.last_updated = chrono::Utc::now();
    }

    /// Update conformal prediction metrics
    pub fn update_conformal_metrics(&mut self, intervals: &ConformalPredictionIntervals) {
        self.conformal_metrics.update(intervals);
        self.metadata.last_updated = chrono::Utc::now();
    }

    /// Update measurement optimization metrics
    pub fn update_measurement_metrics(&mut self, measurements: &OptimizedMeasurements) {
        self.measurement_metrics.update(measurements);
        self.metadata.last_updated = chrono::Utc::now();
    }

    /// Update quantum advantage metric
    pub fn update_quantum_advantage(&mut self, advantage: f64) {
        self.quantum_advantage = advantage;
        self.metadata.last_updated = chrono::Utc::now();
    }

    /// Record computation time
    pub fn record_computation_time(&mut self, operation: &str, duration: Duration) {
        self.performance_metrics.record_computation_time(operation, duration);
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, memory_mb: f64) {
        self.performance_metrics.record_memory_usage(memory_mb);
    }

    /// Record circuit fidelity
    pub fn record_circuit_fidelity(&mut self, fidelity: f64) {
        self.fidelity_metrics.record_fidelity(fidelity);
    }

    /// Get comprehensive metrics summary
    pub fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            overall_performance: self.calculate_overall_performance(),
            quantum_advantage: self.quantum_advantage,
            uncertainty_quality: self.uncertainty_metrics.calculate_quality_score(),
            conformal_coverage: self.conformal_metrics.average_coverage,
            measurement_efficiency: self.measurement_metrics.average_efficiency,
            computational_overhead: self.performance_metrics.average_computation_time_ms,
            memory_usage_mb: self.performance_metrics.current_memory_usage_mb,
            circuit_fidelity: self.fidelity_metrics.average_fidelity,
            system_health: self.system_metrics.health_score,
        }
    }

    /// Calculate overall performance score
    fn calculate_overall_performance(&self) -> f64 {
        let weights = [
            (self.uncertainty_metrics.calculate_quality_score(), 0.25),
            (self.conformal_metrics.average_coverage, 0.20),
            (self.measurement_metrics.average_efficiency, 0.20),
            (self.fidelity_metrics.average_fidelity, 0.15),
            (self.system_metrics.health_score, 0.10),
            (self.quantum_advantage.min(2.0) / 2.0, 0.10), // Normalized quantum advantage
        ];

        weights.iter().map(|(score, weight)| score * weight).sum()
    }

    /// Export metrics to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import metrics from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.uncertainty_metrics.reset();
        self.conformal_metrics.reset();
        self.measurement_metrics.reset();
        self.quantum_advantage = 0.0;
        self.performance_metrics.reset();
        self.fidelity_metrics.reset();
        self.feature_metrics.reset();
        self.correlation_metrics.reset();
        self.system_metrics.reset();
        self.metadata = MetricsMetadata::new();
    }

    /// Get health status
    pub fn get_health_status(&self) -> HealthStatus {
        let overall_score = self.calculate_overall_performance();
        
        if overall_score >= 0.8 {
            HealthStatus::Excellent
        } else if overall_score >= 0.6 {
            HealthStatus::Good
        } else if overall_score >= 0.4 {
            HealthStatus::Fair
        } else if overall_score >= 0.2 {
            HealthStatus::Poor
        } else {
            HealthStatus::Critical
        }
    }
}

/// Uncertainty quantification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyMetrics {
    /// Total uncertainty estimates
    pub total_estimates: u64,
    /// Average uncertainty
    pub average_uncertainty: f64,
    /// Uncertainty variance
    pub uncertainty_variance: f64,
    /// Calibration error
    pub calibration_error: f64,
    /// Sharpness score
    pub sharpness_score: f64,
    /// Coverage probability
    pub coverage_probability: f64,
    /// Uncertainty history (last 100 estimates)
    pub uncertainty_history: VecDeque<f64>,
    /// Quality metrics
    pub quality_metrics: UncertaintyQualityMetrics,
}

impl UncertaintyMetrics {
    pub fn new() -> Self {
        Self {
            total_estimates: 0,
            average_uncertainty: 0.0,
            uncertainty_variance: 0.0,
            calibration_error: 0.0,
            sharpness_score: 0.0,
            coverage_probability: 0.0,
            uncertainty_history: VecDeque::with_capacity(100),
            quality_metrics: UncertaintyQualityMetrics::new(),
        }
    }

    pub fn update(&mut self, estimates: &[UncertaintyEstimate]) {
        self.total_estimates += estimates.len() as u64;
        
        let uncertainties: Vec<f64> = estimates.iter().map(|e| e.uncertainty).collect();
        self.average_uncertainty = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
        
        let mean = self.average_uncertainty;
        self.uncertainty_variance = uncertainties.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / uncertainties.len() as f64;

        // Update history
        for &uncertainty in &uncertainties {
            if self.uncertainty_history.len() >= 100 {
                self.uncertainty_history.pop_front();
            }
            self.uncertainty_history.push_back(uncertainty);
        }

        // Update quality metrics
        self.quality_metrics.update(estimates);
        
        // Calculate calibration error and sharpness
        self.calculate_calibration_metrics(estimates);
    }

    fn calculate_calibration_metrics(&mut self, estimates: &[UncertaintyEstimate]) {
        // Simplified calibration error calculation
        self.calibration_error = estimates.iter()
            .map(|e| (e.confidence_interval.1 - e.confidence_interval.0).abs())
            .sum::<f64>() / estimates.len() as f64;

        // Sharpness is inversely related to interval width
        self.sharpness_score = 1.0 / (1.0 + self.calibration_error);
    }

    pub fn calculate_quality_score(&self) -> f64 {
        let calibration_score = 1.0 - self.calibration_error.min(1.0);
        let sharpness_score = self.sharpness_score;
        let coverage_score = self.coverage_probability;
        
        (calibration_score + sharpness_score + coverage_score) / 3.0
    }

    pub fn reset(&mut self) {
        self.total_estimates = 0;
        self.average_uncertainty = 0.0;
        self.uncertainty_variance = 0.0;
        self.calibration_error = 0.0;
        self.sharpness_score = 0.0;
        self.coverage_probability = 0.0;
        self.uncertainty_history.clear();
        self.quality_metrics.reset();
    }
}

/// Uncertainty quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyQualityMetrics {
    /// Reliability score
    pub reliability_score: f64,
    /// Consistency score
    pub consistency_score: f64,
    /// Precision score
    pub precision_score: f64,
    /// Robustness score
    pub robustness_score: f64,
}

impl UncertaintyQualityMetrics {
    pub fn new() -> Self {
        Self {
            reliability_score: 0.0,
            consistency_score: 0.0,
            precision_score: 0.0,
            robustness_score: 0.0,
        }
    }

    pub fn update(&mut self, estimates: &[UncertaintyEstimate]) {
        self.reliability_score = self.calculate_reliability(estimates);
        self.consistency_score = self.calculate_consistency(estimates);
        self.precision_score = self.calculate_precision(estimates);
        self.robustness_score = self.calculate_robustness(estimates);
    }

    fn calculate_reliability(&self, estimates: &[UncertaintyEstimate]) -> f64 {
        // Simplified reliability calculation
        estimates.iter().map(|e| e.quantum_fidelity).sum::<f64>() / estimates.len() as f64
    }

    fn calculate_consistency(&self, estimates: &[UncertaintyEstimate]) -> f64 {
        if estimates.len() < 2 {
            return 1.0;
        }

        let uncertainties: Vec<f64> = estimates.iter().map(|e| e.uncertainty).collect();
        let mean = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
        let variance = uncertainties.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / uncertainties.len() as f64;
        
        1.0 / (1.0 + variance)
    }

    fn calculate_precision(&self, estimates: &[UncertaintyEstimate]) -> f64 {
        // Precision based on confidence interval width
        let avg_width = estimates.iter()
            .map(|e| e.confidence_interval.1 - e.confidence_interval.0)
            .sum::<f64>() / estimates.len() as f64;
        
        1.0 / (1.0 + avg_width)
    }

    fn calculate_robustness(&self, estimates: &[UncertaintyEstimate]) -> f64 {
        // Robustness based on variance stability
        if estimates.len() < 3 {
            return 1.0;
        }

        let variances: Vec<f64> = estimates.iter().map(|e| e.variance).collect();
        let mean_variance = variances.iter().sum::<f64>() / variances.len() as f64;
        let variance_of_variance = variances.iter()
            .map(|x| (x - mean_variance).powi(2))
            .sum::<f64>() / variances.len() as f64;
        
        1.0 / (1.0 + variance_of_variance)
    }

    pub fn reset(&mut self) {
        self.reliability_score = 0.0;
        self.consistency_score = 0.0;
        self.precision_score = 0.0;
        self.robustness_score = 0.0;
    }
}

/// Conformal prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalMetrics {
    /// Total predictions
    pub total_predictions: u64,
    /// Average coverage
    pub average_coverage: f64,
    /// Average interval width
    pub average_interval_width: f64,
    /// Efficiency score
    pub efficiency_score: f64,
    /// Validity rate
    pub validity_rate: f64,
    /// Coverage history
    pub coverage_history: VecDeque<f64>,
    /// Width history
    pub width_history: VecDeque<f64>,
}

impl ConformalMetrics {
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            average_coverage: 0.0,
            average_interval_width: 0.0,
            efficiency_score: 0.0,
            validity_rate: 0.0,
            coverage_history: VecDeque::with_capacity(100),
            width_history: VecDeque::with_capacity(100),
        }
    }

    pub fn update(&mut self, intervals: &ConformalPredictionIntervals) {
        self.total_predictions += 1;
        
        // Update coverage
        let coverage = intervals.quantum_coverage_probability;
        Self::update_average(&mut self.average_coverage, coverage, self.total_predictions);
        
        // Update interval width
        let width = intervals.average_interval_width();
        Self::update_average(&mut self.average_interval_width, width, self.total_predictions);
        
        // Update efficiency
        self.efficiency_score = intervals.quantum_efficiency;
        
        // Update validity rate
        self.validity_rate = intervals.validity_rate();
        
        // Update histories
        Self::add_to_history(&mut self.coverage_history, coverage);
        Self::add_to_history(&mut self.width_history, width);
    }

    fn update_average(current_avg: &mut f64, new_value: f64, count: u64) {
        *current_avg = (*current_avg * (count - 1) as f64 + new_value) / count as f64;
    }

    fn add_to_history(history: &mut VecDeque<f64>, value: f64) {
        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(value);
    }

    pub fn reset(&mut self) {
        self.total_predictions = 0;
        self.average_coverage = 0.0;
        self.average_interval_width = 0.0;
        self.efficiency_score = 0.0;
        self.validity_rate = 0.0;
        self.coverage_history.clear();
        self.width_history.clear();
    }
}

/// Measurement optimization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMetrics {
    /// Total optimizations
    pub total_optimizations: u64,
    /// Average efficiency
    pub average_efficiency: f64,
    /// Average information gain
    pub average_information_gain: f64,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Average optimization time
    pub average_optimization_time_ms: f64,
    /// Quantum advantage in measurement
    pub measurement_quantum_advantage: f64,
}

impl MeasurementMetrics {
    pub fn new() -> Self {
        Self {
            total_optimizations: 0,
            average_efficiency: 0.0,
            average_information_gain: 0.0,
            convergence_rate: 0.0,
            average_optimization_time_ms: 0.0,
            measurement_quantum_advantage: 0.0,
        }
    }

    pub fn update(&mut self, measurements: &OptimizedMeasurements) {
        self.total_optimizations += 1;
        
        let efficiency = measurements.total_efficiency();
        Self::update_average(&mut self.average_efficiency, efficiency, self.total_optimizations);
        
        let info_gain = measurements.total_information();
        Self::update_average(&mut self.average_information_gain, info_gain, self.total_optimizations);
        
        if measurements.convergence_achieved() {
            self.convergence_rate = (self.convergence_rate * (self.total_optimizations - 1) as f64 + 1.0) / self.total_optimizations as f64;
        } else {
            self.convergence_rate = (self.convergence_rate * (self.total_optimizations - 1) as f64) / self.total_optimizations as f64;
        }
        
        self.measurement_quantum_advantage = measurements.quantum_measurement_advantage();
    }

    fn update_average(current_avg: &mut f64, new_value: f64, count: u64) {
        *current_avg = (*current_avg * (count - 1) as f64 + new_value) / count as f64;
    }

    pub fn reset(&mut self) {
        self.total_optimizations = 0;
        self.average_efficiency = 0.0;
        self.average_information_gain = 0.0;
        self.convergence_rate = 0.0;
        self.average_optimization_time_ms = 0.0;
        self.measurement_quantum_advantage = 0.0;
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Computation times for different operations
    pub computation_times: HashMap<String, Vec<f64>>,
    /// Average computation time
    pub average_computation_time_ms: f64,
    /// Memory usage tracking
    pub memory_usage_history: VecDeque<f64>,
    /// Current memory usage
    pub current_memory_usage_mb: f64,
    /// Peak memory usage
    pub peak_memory_usage_mb: f64,
    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f64,
    /// CPU utilization
    pub cpu_utilization_percent: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            computation_times: HashMap::new(),
            average_computation_time_ms: 0.0,
            memory_usage_history: VecDeque::with_capacity(1000),
            current_memory_usage_mb: 0.0,
            peak_memory_usage_mb: 0.0,
            throughput_ops_per_sec: 0.0,
            cpu_utilization_percent: 0.0,
        }
    }

    pub fn record_computation_time(&mut self, operation: &str, duration: Duration) {
        let time_ms = duration.as_millis() as f64;
        
        self.computation_times
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(time_ms);
        
        // Update average
        let all_times: Vec<f64> = self.computation_times.values().flatten().cloned().collect();
        if !all_times.is_empty() {
            self.average_computation_time_ms = all_times.iter().sum::<f64>() / all_times.len() as f64;
        }
    }

    pub fn record_memory_usage(&mut self, memory_mb: f64) {
        self.current_memory_usage_mb = memory_mb;
        self.peak_memory_usage_mb = self.peak_memory_usage_mb.max(memory_mb);
        
        if self.memory_usage_history.len() >= 1000 {
            self.memory_usage_history.pop_front();
        }
        self.memory_usage_history.push_back(memory_mb);
    }

    pub fn get_operation_stats(&self, operation: &str) -> Option<OperationStats> {
        self.computation_times.get(operation).map(|times| {
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let variance = times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / times.len() as f64;
            let std_dev = variance.sqrt();
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            OperationStats {
                operation: operation.to_string(),
                count: times.len(),
                mean_ms: mean,
                std_dev_ms: std_dev,
                min_ms: min,
                max_ms: max,
            }
        })
    }

    pub fn reset(&mut self) {
        self.computation_times.clear();
        self.average_computation_time_ms = 0.0;
        self.memory_usage_history.clear();
        self.current_memory_usage_mb = 0.0;
        self.peak_memory_usage_mb = 0.0;
        self.throughput_ops_per_sec = 0.0;
        self.cpu_utilization_percent = 0.0;
    }
}

/// Operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub operation: String,
    pub count: usize,
    pub mean_ms: f64,
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

/// Circuit fidelity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityMetrics {
    /// Total fidelity measurements
    pub total_measurements: u64,
    /// Average fidelity
    pub average_fidelity: f64,
    /// Minimum fidelity observed
    pub min_fidelity: f64,
    /// Maximum fidelity observed
    pub max_fidelity: f64,
    /// Fidelity variance
    pub fidelity_variance: f64,
    /// Fidelity history
    pub fidelity_history: VecDeque<f64>,
}

impl FidelityMetrics {
    pub fn new() -> Self {
        Self {
            total_measurements: 0,
            average_fidelity: 0.0,
            min_fidelity: 1.0,
            max_fidelity: 0.0,
            fidelity_variance: 0.0,
            fidelity_history: VecDeque::with_capacity(1000),
        }
    }

    pub fn record_fidelity(&mut self, fidelity: f64) {
        self.total_measurements += 1;
        
        // Update average
        self.average_fidelity = (self.average_fidelity * (self.total_measurements - 1) as f64 + fidelity) / self.total_measurements as f64;
        
        // Update min/max
        self.min_fidelity = self.min_fidelity.min(fidelity);
        self.max_fidelity = self.max_fidelity.max(fidelity);
        
        // Update history
        if self.fidelity_history.len() >= 1000 {
            self.fidelity_history.pop_front();
        }
        self.fidelity_history.push_back(fidelity);
        
        // Update variance
        if self.fidelity_history.len() > 1 {
            let mean = self.average_fidelity;
            self.fidelity_variance = self.fidelity_history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.fidelity_history.len() as f64;
        }
    }

    pub fn reset(&mut self) {
        self.total_measurements = 0;
        self.average_fidelity = 0.0;
        self.min_fidelity = 1.0;
        self.max_fidelity = 0.0;
        self.fidelity_variance = 0.0;
        self.fidelity_history.clear();
    }
}

/// Feature extraction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetrics {
    /// Total feature extractions
    pub total_extractions: u64,
    /// Average feature quality
    pub average_feature_quality: f64,
    /// Feature extraction time
    pub average_extraction_time_ms: f64,
    /// Feature importance distribution
    pub feature_importance_stats: FeatureImportanceStats,
}

impl FeatureMetrics {
    pub fn new() -> Self {
        Self {
            total_extractions: 0,
            average_feature_quality: 0.0,
            average_extraction_time_ms: 0.0,
            feature_importance_stats: FeatureImportanceStats::new(),
        }
    }

    pub fn reset(&mut self) {
        self.total_extractions = 0;
        self.average_feature_quality = 0.0;
        self.average_extraction_time_ms = 0.0;
        self.feature_importance_stats.reset();
    }
}

/// Feature importance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceStats {
    pub mean_importance: f64,
    pub std_importance: f64,
    pub max_importance: f64,
    pub min_importance: f64,
}

impl FeatureImportanceStats {
    pub fn new() -> Self {
        Self {
            mean_importance: 0.0,
            std_importance: 0.0,
            max_importance: 0.0,
            min_importance: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.mean_importance = 0.0;
        self.std_importance = 0.0;
        self.max_importance = 0.0;
        self.min_importance = 0.0;
    }
}

/// Correlation analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMetrics {
    /// Total correlation analyses
    pub total_analyses: u64,
    /// Average correlation strength
    pub average_correlation_strength: f64,
    /// Quantum correlation advantage
    pub quantum_correlation_advantage: f64,
    /// Analysis time
    pub average_analysis_time_ms: f64,
}

impl CorrelationMetrics {
    pub fn new() -> Self {
        Self {
            total_analyses: 0,
            average_correlation_strength: 0.0,
            quantum_correlation_advantage: 0.0,
            average_analysis_time_ms: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.total_analyses = 0;
        self.average_correlation_strength = 0.0;
        self.quantum_correlation_advantage = 0.0;
        self.average_analysis_time_ms = 0.0;
    }
}

/// System health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Overall health score
    pub health_score: f64,
    /// Error rate
    pub error_rate: f64,
    /// Uptime
    pub uptime_seconds: u64,
    /// Last error timestamp
    pub last_error: Option<chrono::DateTime<chrono::Utc>>,
    /// System status
    pub status: SystemStatus,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            health_score: 1.0,
            error_rate: 0.0,
            uptime_seconds: 0,
            last_error: None,
            status: SystemStatus::Healthy,
        }
    }

    pub fn reset(&mut self) {
        self.health_score = 1.0;
        self.error_rate = 0.0;
        self.uptime_seconds = 0;
        self.last_error = None;
        self.status = SystemStatus::Healthy;
    }
}

/// System status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    Healthy,
    Warning,
    Error,
    Critical,
}

/// Metrics metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsMetadata {
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// Version
    pub version: String,
    /// Collection period
    pub collection_period_seconds: u64,
}

impl MetricsMetadata {
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            created_at: now,
            last_updated: now,
            version: "1.0.0".to_string(),
            collection_period_seconds: 60,
        }
    }
}

/// Metrics summary for quick overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Overall performance score (0-1)
    pub overall_performance: f64,
    /// Quantum advantage metric
    pub quantum_advantage: f64,
    /// Uncertainty quality score
    pub uncertainty_quality: f64,
    /// Conformal coverage
    pub conformal_coverage: f64,
    /// Measurement efficiency
    pub measurement_efficiency: f64,
    /// Computational overhead (ms)
    pub computational_overhead: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Circuit fidelity
    pub circuit_fidelity: f64,
    /// System health score
    pub system_health: f64,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Excellent => write!(f, "Excellent"),
            HealthStatus::Good => write!(f, "Good"),
            HealthStatus::Fair => write!(f, "Fair"),
            HealthStatus::Poor => write!(f, "Poor"),
            HealthStatus::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_quantum_metrics_creation() {
        let metrics = QuantumMetrics::new();
        assert_eq!(metrics.uncertainty_metrics.total_estimates, 0);
        assert_eq!(metrics.conformal_metrics.total_predictions, 0);
        assert_eq!(metrics.measurement_metrics.total_optimizations, 0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        metrics.record_computation_time("test_operation", Duration::from_millis(100));
        metrics.record_memory_usage(512.0);
        
        assert!(metrics.average_computation_time_ms > 0.0);
        assert_eq!(metrics.current_memory_usage_mb, 512.0);
        assert_eq!(metrics.peak_memory_usage_mb, 512.0);
    }

    #[test]
    fn test_fidelity_metrics() {
        let mut metrics = FidelityMetrics::new();
        
        metrics.record_fidelity(0.95);
        metrics.record_fidelity(0.98);
        metrics.record_fidelity(0.92);
        
        assert_eq!(metrics.total_measurements, 3);
        assert!(metrics.average_fidelity > 0.9);
        assert_eq!(metrics.min_fidelity, 0.92);
        assert_eq!(metrics.max_fidelity, 0.98);
    }

    #[test]
    fn test_uncertainty_quality_metrics() {
        let mut quality_metrics = UncertaintyQualityMetrics::new();
        
        let estimates = vec![
            UncertaintyEstimate {
                uncertainty: 0.1,
                variance: 0.01,
                confidence_interval: (0.05, 0.15),
                circuit_name: "test".to_string(),
                quantum_fidelity: 0.95,
            },
            UncertaintyEstimate {
                uncertainty: 0.12,
                variance: 0.015,
                confidence_interval: (0.07, 0.17),
                circuit_name: "test".to_string(),
                quantum_fidelity: 0.93,
            },
        ];
        
        quality_metrics.update(&estimates);
        
        assert!(quality_metrics.reliability_score > 0.0);
        assert!(quality_metrics.consistency_score > 0.0);
        assert!(quality_metrics.precision_score > 0.0);
        assert!(quality_metrics.robustness_score > 0.0);
    }

    #[test]
    fn test_metrics_summary() {
        let metrics = QuantumMetrics::new();
        let summary = metrics.get_summary();
        
        assert!(summary.overall_performance >= 0.0);
        assert!(summary.overall_performance <= 1.0);
    }

    #[test]
    fn test_health_status() {
        let metrics = QuantumMetrics::new();
        let status = metrics.get_health_status();
        
        // Default metrics should be in good state
        assert!(matches!(status, HealthStatus::Good | HealthStatus::Excellent));
    }

    #[test]
    fn test_metrics_serialization() {
        let metrics = QuantumMetrics::new();
        let json = metrics.to_json().unwrap();
        let deserialized = QuantumMetrics::from_json(&json).unwrap();
        
        assert_eq!(metrics.uncertainty_metrics.total_estimates, deserialized.uncertainty_metrics.total_estimates);
    }

    #[test]
    fn test_operation_stats() {
        let mut metrics = PerformanceMetrics::new();
        
        metrics.record_computation_time("test_op", Duration::from_millis(50));
        metrics.record_computation_time("test_op", Duration::from_millis(100));
        metrics.record_computation_time("test_op", Duration::from_millis(75));
        
        let stats = metrics.get_operation_stats("test_op").unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean_ms, 75.0);
        assert_eq!(stats.min_ms, 50.0);
        assert_eq!(stats.max_ms, 100.0);
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = QuantumMetrics::new();
        
        // Add some data
        metrics.record_computation_time("test", Duration::from_millis(100));
        metrics.record_memory_usage(512.0);
        metrics.record_circuit_fidelity(0.95);
        
        // Reset
        metrics.reset();
        
        assert_eq!(metrics.performance_metrics.average_computation_time_ms, 0.0);
        assert_eq!(metrics.performance_metrics.current_memory_usage_mb, 0.0);
        assert_eq!(metrics.fidelity_metrics.total_measurements, 0);
    }
}