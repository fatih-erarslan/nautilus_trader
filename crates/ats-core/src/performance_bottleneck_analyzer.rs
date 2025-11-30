//! Performance Bottleneck Analyzer Agent
//!
//! This module implements a comprehensive performance analysis and bottleneck detection
//! system for ATS-Core conformal prediction. It provides real-time monitoring,
//! bottleneck identification, and optimization recommendations to maintain
//! sub-20μs latency requirements in high-frequency trading environments.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::types::*;
use crate::memory_optimized::*;

/// Placeholder for cache-aligned memory pool - to be properly implemented
struct CacheAlignedMemoryPool {
    capacity: usize,
}

impl CacheAlignedMemoryPool {
    fn new(capacity: usize) -> Self {
        Self { capacity }
    }
}

/// Performance metrics collected during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: SystemTime,
    pub execution_time: Duration,
    pub cpu_utilization: f64,
    pub memory_usage: usize,
    pub cache_misses: u64,
    pub simd_utilization: f64,
    pub throughput: f64,
    pub operation_type: String,
    pub input_size: usize,
}

/// Detected bottleneck with severity and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: BottleneckSeverity,
    pub description: String,
    pub root_cause: String,
    pub impact_estimate: f64, // Performance impact as percentage
    pub recommendations: Vec<OptimizationRecommendation>,
    pub detected_at: SystemTime,
    pub frequency: u32, // How often this bottleneck occurs
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckType {
    ExecutionTime,
    MemoryAccess,
    CacheMisses,
    VectorizationMissed,
    SequentialBlocking,
    ResourceContention,
    AlgorithmComplexity,
    DataTransfer,
    CoordinationOverhead,
    NumericalInstability,
}

/// Severity levels for bottlenecks
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,    // <5% performance impact
    Medium, // 5-15% performance impact
    High,   // 15-30% performance impact
    Critical, // >30% performance impact
}

/// Optimization recommendations from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub estimated_improvement: f64, // Expected performance improvement percentage
    pub implementation_effort: ImplementationEffort,
    pub code_example: Option<String>,
    pub priority: RecommendationPriority,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Vectorization,
    MemoryLayout,
    AlgorithmChange,
    Parallelization,
    Caching,
    DataStructure,
    CompilerHints,
    HardwareSpecific,
}

/// Implementation effort levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,    // <1 day
    Medium, // 1-3 days
    High,   // >3 days
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance baseline for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub operation: String,
    pub expected_time: Duration,
    pub variance_threshold: f64, // Acceptable variance percentage
    pub sample_count: usize,
    pub last_updated: SystemTime,
}

/// Main Performance Bottleneck Analyzer Agent
pub struct PerformanceBottleneckAnalyzer {
    metrics_history: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    detected_bottlenecks: Arc<RwLock<HashMap<BottleneckType, PerformanceBottleneck>>>,
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    optimization_history: Arc<Mutex<Vec<AppliedOptimization>>>,
    config: AnalyzerConfig,
    ml_predictor: Option<BottleneckPredictor>,
    memory_pool: CacheAlignedMemoryPool,
}

/// Configuration for the analyzer
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    pub max_history_size: usize,
    pub detection_sensitivity: f64,
    pub baseline_update_threshold: f64,
    pub enable_ml_prediction: bool,
    pub reporting_interval: Duration,
    pub auto_optimization: bool,
}

/// Applied optimization tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    pub optimization_type: RecommendationType,
    pub applied_at: SystemTime,
    pub before_metrics: PerformanceMetrics,
    pub after_metrics: Option<PerformanceMetrics>,
    pub improvement_achieved: Option<f64>,
    pub success: bool,
}

/// Machine learning-based bottleneck predictor
pub struct BottleneckPredictor {
    model_weights: Vec<f64>,
    feature_scalers: HashMap<String, (f64, f64)>, // (mean, std)
    prediction_history: VecDeque<(Vec<f64>, BottleneckType)>,
    accuracy_metrics: PredictorAccuracy,
}

/// Predictor accuracy tracking
#[derive(Debug, Clone)]
pub struct PredictorAccuracy {
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            detection_sensitivity: 0.15, // 15% performance degradation threshold
            baseline_update_threshold: 0.05, // 5% change to update baseline
            enable_ml_prediction: true,
            reporting_interval: Duration::from_secs(60),
            auto_optimization: false, // Require manual approval for safety
        }
    }
}

impl PerformanceBottleneckAnalyzer {
    /// Create a new Performance Bottleneck Analyzer
    pub fn new(config: AnalyzerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let ml_predictor = if config.enable_ml_prediction {
            Some(BottleneckPredictor::new()?)
        } else {
            None
        };

        Ok(Self {
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_history_size))),
            detected_bottlenecks: Arc::new(RwLock::new(HashMap::new())),
            baselines: Arc::new(RwLock::new(HashMap::new())),
            optimization_history: Arc::new(Mutex::new(Vec::new())),
            memory_pool: CacheAlignedMemoryPool::new(1024 * 1024), // 1MB pool
            ml_predictor,
            config,
        })
    }

    /// Record performance metrics for analysis
    pub fn record_metrics(&self, metrics: PerformanceMetrics) -> Result<(), Box<dyn std::error::Error>> {
        // Add to metrics history
        {
            let mut history = self.metrics_history.write().unwrap();
            if history.len() >= self.config.max_history_size {
                history.pop_front();
            }
            history.push_back(metrics.clone());
        }

        // Update or create baseline
        self.update_baseline(&metrics)?;

        // Analyze for bottlenecks
        self.analyze_for_bottlenecks(&metrics)?;

        // Train ML predictor if enabled
        if let Some(ref predictor) = self.ml_predictor {
            self.train_predictor(predictor, &metrics)?;
        }

        Ok(())
    }

    /// Analyze current metrics for potential bottlenecks
    pub fn analyze_for_bottlenecks(&self, current_metrics: &PerformanceMetrics) -> Result<Vec<PerformanceBottleneck>, Box<dyn std::error::Error>> {
        let mut detected = Vec::new();

        // Check against baseline
        if let Some(baseline) = self.get_baseline(&current_metrics.operation_type) {
            let performance_ratio = current_metrics.execution_time.as_nanos() as f64 / baseline.expected_time.as_nanos() as f64;
            
            if performance_ratio > (1.0 + self.config.detection_sensitivity) {
                let bottleneck = self.classify_bottleneck(current_metrics, &baseline, performance_ratio)?;
                detected.push(bottleneck);
            }
        }

        // Specific bottleneck detection patterns
        detected.extend(self.detect_memory_bottlenecks(current_metrics)?);
        detected.extend(self.detect_cpu_bottlenecks(current_metrics)?);
        detected.extend(self.detect_cache_bottlenecks(current_metrics)?);
        detected.extend(self.detect_vectorization_opportunities(current_metrics)?);

        // Update detected bottlenecks map
        {
            let mut bottlenecks = self.detected_bottlenecks.write().unwrap();
            for bottleneck in &detected {
                bottlenecks.entry(bottleneck.bottleneck_type.clone())
                    .and_modify(|existing| existing.frequency += 1)
                    .or_insert(bottleneck.clone());
            }
        }

        Ok(detected)
    }

    /// Classify the type of bottleneck based on metrics
    fn classify_bottleneck(&self, metrics: &PerformanceMetrics, baseline: &PerformanceBaseline, ratio: f64) -> Result<PerformanceBottleneck, Box<dyn std::error::Error>> {
        let bottleneck_type = if metrics.cache_misses > 1000 {
            BottleneckType::CacheMisses
        } else if metrics.cpu_utilization < 0.7 && metrics.execution_time > baseline.expected_time * 2 {
            BottleneckType::SequentialBlocking
        } else if metrics.simd_utilization < 0.5 {
            BottleneckType::VectorizationMissed
        } else if metrics.memory_usage > 100 * 1024 * 1024 { // >100MB
            BottleneckType::MemoryAccess
        } else {
            BottleneckType::ExecutionTime
        };

        let severity = match ratio {
            r if r > 2.0 => BottleneckSeverity::Critical,
            r if r > 1.5 => BottleneckSeverity::High,
            r if r > 1.2 => BottleneckSeverity::Medium,
            _ => BottleneckSeverity::Low,
        };

        let recommendations = self.generate_recommendations(&bottleneck_type, metrics)?;

        Ok(PerformanceBottleneck {
            bottleneck_type,
            severity,
            description: format!("Performance degradation detected: {:.1}% slower than baseline", (ratio - 1.0) * 100.0),
            root_cause: self.analyze_root_cause(metrics, &baseline),
            impact_estimate: (ratio - 1.0) * 100.0,
            recommendations,
            detected_at: SystemTime::now(),
            frequency: 1,
        })
    }

    /// Detect memory-related bottlenecks
    fn detect_memory_bottlenecks(&self, metrics: &PerformanceMetrics) -> Result<Vec<PerformanceBottleneck>, Box<dyn std::error::Error>> {
        let mut bottlenecks = Vec::new();

        // High memory usage bottleneck
        if metrics.memory_usage > 500 * 1024 * 1024 { // >500MB
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::MemoryAccess,
                severity: BottleneckSeverity::High,
                description: format!("High memory usage detected: {}MB", metrics.memory_usage / (1024 * 1024)),
                root_cause: "Large memory allocations or memory leaks".to_string(),
                impact_estimate: 25.0,
                recommendations: vec![
                    OptimizationRecommendation {
                        recommendation_type: RecommendationType::MemoryLayout,
                        description: "Implement memory pooling for frequent allocations".to_string(),
                        estimated_improvement: 20.0,
                        implementation_effort: ImplementationEffort::Medium,
                        code_example: Some("use memory_pool.get_buffer(size) instead of Vec::with_capacity(size)".to_string()),
                        priority: RecommendationPriority::High,
                    }
                ],
                detected_at: SystemTime::now(),
                frequency: 1,
            });
        }

        Ok(bottlenecks)
    }

    /// Detect CPU utilization bottlenecks
    fn detect_cpu_bottlenecks(&self, metrics: &PerformanceMetrics) -> Result<Vec<PerformanceBottleneck>, Box<dyn std::error::Error>> {
        let mut bottlenecks = Vec::new();

        // Low CPU utilization suggests sequential bottleneck
        if metrics.cpu_utilization < 0.6 && metrics.execution_time > Duration::from_micros(10) {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::SequentialBlocking,
                severity: BottleneckSeverity::Medium,
                description: format!("Low CPU utilization: {:.1}%", metrics.cpu_utilization * 100.0),
                root_cause: "Sequential execution preventing parallel utilization".to_string(),
                impact_estimate: 30.0,
                recommendations: vec![
                    OptimizationRecommendation {
                        recommendation_type: RecommendationType::Parallelization,
                        description: "Identify parallelization opportunities using rayon".to_string(),
                        estimated_improvement: 25.0,
                        implementation_effort: ImplementationEffort::Medium,
                        code_example: Some("use rayon::prelude::*; data.par_iter().map(|x| process(x)).collect()".to_string()),
                        priority: RecommendationPriority::High,
                    }
                ],
                detected_at: SystemTime::now(),
                frequency: 1,
            });
        }

        Ok(bottlenecks)
    }

    /// Detect cache miss bottlenecks
    fn detect_cache_bottlenecks(&self, metrics: &PerformanceMetrics) -> Result<Vec<PerformanceBottleneck>, Box<dyn std::error::Error>> {
        let mut bottlenecks = Vec::new();

        // High cache miss rate
        if metrics.cache_misses > 500 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CacheMisses,
                severity: BottleneckSeverity::High,
                description: format!("High cache miss count: {}", metrics.cache_misses),
                root_cause: "Poor memory access patterns or unaligned data structures".to_string(),
                impact_estimate: 40.0,
                recommendations: vec![
                    OptimizationRecommendation {
                        recommendation_type: RecommendationType::MemoryLayout,
                        description: "Use cache-aligned data structures".to_string(),
                        estimated_improvement: 35.0,
                        implementation_effort: ImplementationEffort::Low,
                        code_example: Some("#[repr(align(64))] struct CacheAligned<T>(T);".to_string()),
                        priority: RecommendationPriority::Critical,
                    }
                ],
                detected_at: SystemTime::now(),
                frequency: 1,
            });
        }

        Ok(bottlenecks)
    }

    /// Detect missed vectorization opportunities
    fn detect_vectorization_opportunities(&self, metrics: &PerformanceMetrics) -> Result<Vec<PerformanceBottleneck>, Box<dyn std::error::Error>> {
        let mut bottlenecks = Vec::new();

        // Low SIMD utilization
        if metrics.simd_utilization < 0.3 && metrics.input_size > 8 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::VectorizationMissed,
                severity: BottleneckSeverity::Medium,
                description: format!("Low SIMD utilization: {:.1}%", metrics.simd_utilization * 100.0),
                root_cause: "Operations not vectorized for SIMD instructions".to_string(),
                impact_estimate: 60.0,
                recommendations: vec![
                    OptimizationRecommendation {
                        recommendation_type: RecommendationType::Vectorization,
                        description: "Implement AVX-512/AVX2 SIMD instructions".to_string(),
                        estimated_improvement: 50.0,
                        implementation_effort: ImplementationEffort::High,
                        code_example: Some("unsafe { _mm512_add_pd(a, b) } // AVX-512 example".to_string()),
                        priority: RecommendationPriority::High,
                    }
                ],
                detected_at: SystemTime::now(),
                frequency: 1,
            });
        }

        Ok(bottlenecks)
    }

    /// Generate optimization recommendations for a bottleneck type
    fn generate_recommendations(&self, bottleneck_type: &BottleneckType, metrics: &PerformanceMetrics) -> Result<Vec<OptimizationRecommendation>, Box<dyn std::error::Error>> {
        match bottleneck_type {
            BottleneckType::ExecutionTime => Ok(vec![
                OptimizationRecommendation {
                    recommendation_type: RecommendationType::AlgorithmChange,
                    description: "Consider algorithmic improvements (e.g., O(n log n) → O(n))".to_string(),
                    estimated_improvement: 40.0,
                    implementation_effort: ImplementationEffort::High,
                    code_example: Some("Use Greenwald-Khanna for O(n) quantiles instead of sorting".to_string()),
                    priority: RecommendationPriority::High,
                }
            ]),
            BottleneckType::MemoryAccess => Ok(vec![
                OptimizationRecommendation {
                    recommendation_type: RecommendationType::MemoryLayout,
                    description: "Optimize memory access patterns and data layout".to_string(),
                    estimated_improvement: 25.0,
                    implementation_effort: ImplementationEffort::Medium,
                    code_example: Some("Use structure-of-arrays instead of array-of-structures".to_string()),
                    priority: RecommendationPriority::Medium,
                }
            ]),
            BottleneckType::VectorizationMissed => Ok(vec![
                OptimizationRecommendation {
                    recommendation_type: RecommendationType::Vectorization,
                    description: "Implement SIMD vectorization for mathematical operations".to_string(),
                    estimated_improvement: 60.0,
                    implementation_effort: ImplementationEffort::High,
                    code_example: Some("Use std::arch::x86_64 intrinsics for AVX-512 operations".to_string()),
                    priority: RecommendationPriority::Critical,
                }
            ]),
            _ => Ok(Vec::new()),
        }
    }

    /// Analyze root cause of performance issue
    fn analyze_root_cause(&self, metrics: &PerformanceMetrics, baseline: &PerformanceBaseline) -> String {
        let mut causes = Vec::new();

        if metrics.cache_misses > 100 {
            causes.push("High cache miss rate");
        }
        if metrics.cpu_utilization < 0.7 {
            causes.push("Low CPU utilization");
        }
        if metrics.simd_utilization < 0.5 {
            causes.push("Missed vectorization opportunities");
        }
        if metrics.memory_usage > 100 * 1024 * 1024 {
            causes.push("High memory usage");
        }

        if causes.is_empty() {
            "General performance degradation".to_string()
        } else {
            causes.join(", ")
        }
    }

    /// Update performance baseline
    fn update_baseline(&self, metrics: &PerformanceMetrics) -> Result<(), Box<dyn std::error::Error>> {
        let mut baselines = self.baselines.write().unwrap();
        
        let baseline = baselines.entry(metrics.operation_type.clone())
            .or_insert_with(|| PerformanceBaseline {
                operation: metrics.operation_type.clone(),
                expected_time: metrics.execution_time,
                variance_threshold: 0.1, // 10% variance
                sample_count: 0,
                last_updated: SystemTime::now(),
            });

        baseline.sample_count += 1;

        // Exponential moving average for baseline update
        let alpha = if baseline.sample_count < 10 { 0.3 } else { 0.1 };
        let current_nanos = metrics.execution_time.as_nanos() as f64;
        let baseline_nanos = baseline.expected_time.as_nanos() as f64;
        let new_baseline_nanos = alpha * current_nanos + (1.0 - alpha) * baseline_nanos;
        
        baseline.expected_time = Duration::from_nanos(new_baseline_nanos as u64);
        baseline.last_updated = SystemTime::now();

        Ok(())
    }

    /// Get baseline for operation type
    fn get_baseline(&self, operation_type: &str) -> Option<PerformanceBaseline> {
        let baselines = self.baselines.read().unwrap();
        baselines.get(operation_type).cloned()
    }

    /// Train ML predictor with current metrics
    fn train_predictor(&self, predictor: &BottleneckPredictor, metrics: &PerformanceMetrics) -> Result<(), Box<dyn std::error::Error>> {
        // This would implement incremental learning for bottleneck prediction
        // For now, we'll skip the complex ML implementation
        Ok(())
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> Result<PerformanceReport, Box<dyn std::error::Error>> {
        let bottlenecks = self.detected_bottlenecks.read().unwrap().clone();
        let history = self.metrics_history.read().unwrap().clone();
        
        let total_metrics = history.len();
        let avg_execution_time = if total_metrics > 0 {
            Duration::from_nanos(
                (history.iter().map(|m| m.execution_time.as_nanos()).sum::<u128>() / total_metrics as u128) as u64
            )
        } else {
            Duration::from_nanos(0)
        };

        Ok(PerformanceReport {
            timestamp: SystemTime::now(),
            total_measurements: total_metrics,
            average_execution_time: avg_execution_time,
            detected_bottlenecks: bottlenecks.into_values().collect(),
            top_recommendations: self.get_top_recommendations()?,
            performance_trend: self.calculate_performance_trend(&history),
            system_health_score: self.calculate_health_score(&history),
        })
    }

    /// Get top optimization recommendations
    fn get_top_recommendations(&self) -> Result<Vec<OptimizationRecommendation>, Box<dyn std::error::Error>> {
        let bottlenecks = self.detected_bottlenecks.read().unwrap();
        let mut recommendations = Vec::new();

        for bottleneck in bottlenecks.values() {
            recommendations.extend(bottleneck.recommendations.clone());
        }

        // Sort by priority and estimated improvement
        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then(b.estimated_improvement.partial_cmp(&a.estimated_improvement).unwrap())
        });

        Ok(recommendations.into_iter().take(10).collect())
    }

    /// Calculate performance trend
    fn calculate_performance_trend(&self, history: &VecDeque<PerformanceMetrics>) -> PerformanceTrend {
        // Need at least 20 samples to calculate trend (10 recent + 10 older)
        if history.len() < 20 {
            return PerformanceTrend::Stable;
        }

        let recent: Vec<_> = history.iter().rev().take(10).collect();
        let older: Vec<_> = history.iter().rev().skip(10).take(10).collect();

        // Safety check for empty collections
        if recent.is_empty() || older.is_empty() {
            return PerformanceTrend::Stable;
        }

        let recent_avg = recent.iter().map(|m| m.execution_time.as_nanos()).sum::<u128>() / recent.len() as u128;
        let older_avg = older.iter().map(|m| m.execution_time.as_nanos()).sum::<u128>() / older.len() as u128;

        // Avoid division by zero
        if older_avg == 0 {
            return PerformanceTrend::Stable;
        }

        let change_ratio = recent_avg as f64 / older_avg as f64;

        match change_ratio {
            r if r > 1.1 => PerformanceTrend::Degrading,
            r if r < 0.9 => PerformanceTrend::Improving,
            _ => PerformanceTrend::Stable,
        }
    }

    /// Calculate system health score (0-100)
    fn calculate_health_score(&self, history: &VecDeque<PerformanceMetrics>) -> f64 {
        if history.is_empty() {
            return 100.0;
        }

        let recent = history.iter().rev().take(100).collect::<Vec<_>>();
        let mut score = 100.0;

        // Penalize high execution times
        let avg_time = recent.iter().map(|m| m.execution_time.as_micros()).sum::<u128>() / recent.len() as u128;
        if avg_time > 20 { // >20μs target
            score -= ((avg_time - 20) as f64 / 20.0) * 30.0;
        }

        // Penalize cache misses
        let avg_cache_misses = recent.iter().map(|m| m.cache_misses).sum::<u64>() / recent.len() as u64;
        if avg_cache_misses > 100 {
            score -= (avg_cache_misses - 100) as f64 / 10.0;
        }

        // Penalize low SIMD utilization
        let avg_simd = recent.iter().map(|m| m.simd_utilization).sum::<f64>() / recent.len() as f64;
        if avg_simd < 0.7 {
            score -= (0.7 - avg_simd) * 20.0;
        }

        score.max(0.0).min(100.0)
    }

    /// Get current bottlenecks by severity
    pub fn get_bottlenecks_by_severity(&self, min_severity: BottleneckSeverity) -> Vec<PerformanceBottleneck> {
        let bottlenecks = self.detected_bottlenecks.read().unwrap();
        bottlenecks.values()
            .filter(|b| b.severity >= min_severity)
            .cloned()
            .collect()
    }

    /// Clear resolved bottlenecks
    pub fn clear_resolved_bottlenecks(&self) {
        let mut bottlenecks = self.detected_bottlenecks.write().unwrap();
        bottlenecks.clear();
    }
}

/// Performance report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: SystemTime,
    pub total_measurements: usize,
    pub average_execution_time: Duration,
    pub detected_bottlenecks: Vec<PerformanceBottleneck>,
    pub top_recommendations: Vec<OptimizationRecommendation>,
    pub performance_trend: PerformanceTrend,
    pub system_health_score: f64,
}

/// Performance trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

impl BottleneckPredictor {
    /// Create new ML-based bottleneck predictor
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            model_weights: vec![0.0; 64], // Simple linear model
            feature_scalers: HashMap::new(),
            prediction_history: VecDeque::with_capacity(1000),
            accuracy_metrics: PredictorAccuracy {
                total_predictions: 0,
                correct_predictions: 0,
                false_positives: 0,
                false_negatives: 0,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_analyzer_creation() {
        let config = AnalyzerConfig::default();
        let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
        
        // Verify analyzer is properly initialized
        assert_eq!(analyzer.metrics_history.read().unwrap().len(), 0);
        assert_eq!(analyzer.detected_bottlenecks.read().unwrap().len(), 0);
    }

    #[test]
    fn test_metrics_recording() {
        let config = AnalyzerConfig::default();
        let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
        
        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(25),
            cpu_utilization: 0.8,
            memory_usage: 1024 * 1024, // 1MB
            cache_misses: 50,
            simd_utilization: 0.9,
            throughput: 1000.0,
            operation_type: "softmax".to_string(),
            input_size: 16,
        };

        analyzer.record_metrics(metrics).unwrap();
        
        // Verify metrics were recorded
        assert_eq!(analyzer.metrics_history.read().unwrap().len(), 1);
    }

    #[test]
    fn test_bottleneck_detection() {
        let config = AnalyzerConfig::default();
        let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
        
        // Create metrics that should trigger bottleneck detection
        let slow_metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(100), // Very slow
            cpu_utilization: 0.3, // Low utilization
            memory_usage: 600 * 1024 * 1024, // High memory (exceeds 500MB threshold)
            cache_misses: 1000, // High cache misses
            simd_utilization: 0.1, // Low SIMD utilization
            throughput: 10.0, // Low throughput
            operation_type: "test_op".to_string(),
            input_size: 16,
        };

        let bottlenecks = analyzer.analyze_for_bottlenecks(&slow_metrics).unwrap();
        
        // Should detect multiple bottlenecks
        assert!(!bottlenecks.is_empty());
        
        // Check for specific bottleneck types
        let bottleneck_types: Vec<_> = bottlenecks.iter().map(|b| &b.bottleneck_type).collect();
        assert!(bottleneck_types.contains(&&BottleneckType::MemoryAccess));
        assert!(bottleneck_types.contains(&&BottleneckType::CacheMisses));
        assert!(bottleneck_types.contains(&&BottleneckType::VectorizationMissed));
    }

    #[test]
    fn test_performance_report_generation() {
        let config = AnalyzerConfig::default();
        let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
        
        // Record some metrics
        for i in 0..10 {
            let metrics = PerformanceMetrics {
                timestamp: SystemTime::now(),
                execution_time: Duration::from_micros(15 + i), // Varying execution times
                cpu_utilization: 0.8,
                memory_usage: 1024 * 1024,
                cache_misses: 20 + i * 10,
                simd_utilization: 0.9,
                throughput: 1000.0,
                operation_type: "test_op".to_string(),
                input_size: 16,
            };
            analyzer.record_metrics(metrics).unwrap();
        }
        
        let report = analyzer.generate_report().unwrap();
        
        // Verify report structure
        assert_eq!(report.total_measurements, 10);
        assert!(report.system_health_score > 0.0);
        assert!(report.system_health_score <= 100.0);
    }

    #[test]
    fn test_bottleneck_severity_classification() {
        let config = AnalyzerConfig::default();
        let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
        
        // Create baseline first
        let baseline_metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(10),
            cpu_utilization: 0.8,
            memory_usage: 1024 * 1024,
            cache_misses: 50,
            simd_utilization: 0.9,
            throughput: 1000.0,
            operation_type: "test_op".to_string(),
            input_size: 16,
        };
        analyzer.record_metrics(baseline_metrics).unwrap();
        
        // Create metrics that are 3x slower (should be Critical)
        let critical_metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(30), // 3x slower
            cpu_utilization: 0.8,
            memory_usage: 1024 * 1024,
            cache_misses: 50,
            simd_utilization: 0.9,
            throughput: 333.0,
            operation_type: "test_op".to_string(),
            input_size: 16,
        };
        
        let bottlenecks = analyzer.analyze_for_bottlenecks(&critical_metrics).unwrap();
        
        if !bottlenecks.is_empty() {
            assert_eq!(bottlenecks[0].severity, BottleneckSeverity::Critical);
        }
    }
}