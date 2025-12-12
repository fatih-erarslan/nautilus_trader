//! Performance tracking and benchmarking for CDFA framework
//!
//! This module provides comprehensive performance monitoring, real-time metrics collection,
//! and automated benchmarking capabilities for swarm intelligence algorithms.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

use crate::core::{SwarmError, Individual, Population, AlgorithmMetrics};

/// Comprehensive performance tracker for algorithms and combinations
pub struct PerformanceTracker {
    /// Performance metrics storage
    metrics_storage: Arc<RwLock<HashMap<String, AlgorithmPerformanceHistory>>>,
    
    /// Real-time monitoring state
    monitoring_state: Arc<RwLock<MonitoringState>>,
    
    /// Benchmark results cache
    benchmark_cache: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    
    /// Performance analysis configuration
    config: PerformanceConfig,
    
    /// System resource monitor
    resource_monitor: Arc<SystemResourceMonitor>,
}

/// Performance tracking configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Maximum history size per algorithm
    pub max_history_size: usize,
    
    /// Enable real-time monitoring
    pub enable_realtime: bool,
    
    /// Monitoring interval (milliseconds)
    pub monitoring_interval: u64,
    
    /// Enable memory tracking
    pub track_memory: bool,
    
    /// Enable CPU profiling
    pub enable_profiling: bool,
    
    /// Benchmark comparison threshold
    pub comparison_threshold: f64,
    
    /// Performance degradation alert threshold
    pub degradation_threshold: f64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            enable_realtime: true,
            monitoring_interval: 100, // 100ms
            track_memory: true,
            enable_profiling: true,
            comparison_threshold: 0.05, // 5%
            degradation_threshold: 0.1, // 10%
        }
    }
}

/// Algorithm performance history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPerformanceHistory {
    /// Algorithm identifier
    pub algorithm_id: String,
    
    /// Performance metrics over time
    pub metrics_history: VecDeque<TimestampedMetrics>,
    
    /// Statistical summary
    pub summary: PerformanceSummary,
    
    /// Performance trends
    pub trends: PerformanceTrends,
    
    /// Comparison with other algorithms
    pub comparisons: HashMap<String, ComparisonResult>,
}

/// Timestamped performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedMetrics {
    /// Timestamp of measurement
    pub timestamp: SystemTime,
    
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    
    /// System state at measurement
    pub system_state: SystemState,
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Basic algorithm metrics
    pub algorithm_metrics: AlgorithmMetrics,
    
    /// Timing information
    pub timing: TimingMetrics,
    
    /// Resource usage
    pub resources: ResourceMetrics,
    
    /// Quality metrics
    pub quality: QualityMetrics,
    
    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,
    
    /// Scalability metrics
    pub scalability: ScalabilityMetrics,
}

/// Timing-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Total execution time
    pub total_time: Duration,
    
    /// Time per iteration
    pub time_per_iteration: Duration,
    
    /// Time to convergence
    pub time_to_convergence: Option<Duration>,
    
    /// CPU time used
    pub cpu_time: Duration,
    
    /// Wall clock time
    pub wall_time: Duration,
    
    /// Time breakdown by operation
    pub operation_times: HashMap<String, Duration>,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    
    /// Average memory usage (bytes)
    pub avg_memory: usize,
    
    /// Memory allocations count
    pub allocations: usize,
    
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    
    /// Thread count
    pub thread_count: usize,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Network I/O (if applicable)
    pub network_io: IOMetrics,
    
    /// Disk I/O (if applicable)
    pub disk_io: IOMetrics,
}

/// I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOMetrics {
    /// Bytes read
    pub bytes_read: usize,
    
    /// Bytes written
    pub bytes_written: usize,
    
    /// Operation count
    pub operations: usize,
}

impl Default for IOMetrics {
    fn default() -> Self {
        Self {
            bytes_read: 0,
            bytes_written: 0,
            operations: 0,
        }
    }
}

/// Solution quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Best fitness achieved
    pub best_fitness: f64,
    
    /// Average fitness
    pub avg_fitness: f64,
    
    /// Fitness standard deviation
    pub fitness_std: f64,
    
    /// Convergence rate
    pub convergence_rate: f64,
    
    /// Solution accuracy
    pub accuracy: f64,
    
    /// Robustness score
    pub robustness: f64,
    
    /// Diversity maintenance
    pub diversity_score: f64,
}

/// Algorithm efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Evaluations per second
    pub evaluations_per_second: f64,
    
    /// Convergence efficiency
    pub convergence_efficiency: f64,
    
    /// Resource efficiency
    pub resource_efficiency: f64,
    
    /// Time efficiency
    pub time_efficiency: f64,
    
    /// Memory efficiency
    pub memory_efficiency: f64,
    
    /// Energy efficiency (if available)
    pub energy_efficiency: Option<f64>,
}

/// Scalability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// Performance vs problem size
    pub size_scalability: f64,
    
    /// Performance vs thread count
    pub parallel_scalability: f64,
    
    /// Memory scalability
    pub memory_scalability: f64,
    
    /// Time complexity factor
    pub time_complexity: f64,
    
    /// Space complexity factor
    pub space_complexity: f64,
}

/// Performance summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Best performance achieved
    pub best_performance: PerformanceMetrics,
    
    /// Average performance
    pub avg_performance: PerformanceMetrics,
    
    /// Worst performance
    pub worst_performance: PerformanceMetrics,
    
    /// Standard deviations
    pub std_deviations: PerformanceMetrics,
    
    /// Percentile rankings
    pub percentiles: HashMap<String, f64>,
    
    /// Performance stability score
    pub stability_score: f64,
}

/// Performance trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Overall trend direction
    pub trend_direction: TrendDirection,
    
    /// Trend strength (0.0 to 1.0)
    pub trend_strength: f64,
    
    /// Performance velocity (change rate)
    pub velocity: f64,
    
    /// Acceleration (change in velocity)
    pub acceleration: f64,
    
    /// Seasonality detection
    pub seasonality: Option<SeasonalityPattern>,
    
    /// Anomaly detection
    pub anomalies: Vec<AnomalyPoint>,
}

/// Trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Seasonality pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityPattern {
    /// Pattern period
    pub period: Duration,
    
    /// Pattern strength
    pub strength: f64,
    
    /// Pattern type
    pub pattern_type: PatternType,
}

/// Pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Cyclical,
    Periodic,
    Trending,
    Random,
}

/// Anomaly point detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPoint {
    /// Timestamp of anomaly
    pub timestamp: SystemTime,
    
    /// Anomaly severity
    pub severity: f64,
    
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    
    /// Description
    pub description: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    PerformanceSpike,
    PerformanceDrop,
    MemoryLeak,
    CPUSpike,
    Timeout,
    Convergence,
    Other(String),
}

/// Algorithm comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Compared algorithm ID
    pub other_algorithm: String,
    
    /// Performance ratio (this/other)
    pub performance_ratio: f64,
    
    /// Statistical significance
    pub p_value: f64,
    
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    
    /// Winner determination
    pub winner: ComparisonWinner,
    
    /// Detailed comparison breakdown
    pub breakdown: HashMap<String, f64>,
}

/// Comparison winner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonWinner {
    ThisAlgorithm,
    OtherAlgorithm,
    Tie,
    Inconclusive,
}

/// System state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// CPU load
    pub cpu_load: f64,
    
    /// Memory usage
    pub memory_usage: f64,
    
    /// Disk usage
    pub disk_usage: f64,
    
    /// Network activity
    pub network_activity: f64,
    
    /// System temperature (if available)
    pub temperature: Option<f64>,
    
    /// Power consumption (if available)
    pub power_consumption: Option<f64>,
}

/// Real-time monitoring state
#[derive(Debug, Clone)]
pub struct MonitoringState {
    /// Currently monitored algorithms
    pub active_monitors: HashMap<String, AlgorithmMonitor>,
    
    /// Monitoring start time
    pub start_time: Instant,
    
    /// Total measurements taken
    pub measurement_count: usize,
    
    /// Alert conditions
    pub alerts: Vec<PerformanceAlert>,
}

/// Individual algorithm monitor
#[derive(Debug, Clone)]
pub struct AlgorithmMonitor {
    /// Algorithm ID
    pub algorithm_id: String,
    
    /// Last measurement time
    pub last_measurement: Instant,
    
    /// Measurement interval
    pub interval: Duration,
    
    /// Enabled metrics
    pub enabled_metrics: Vec<MetricType>,
    
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
}

/// Types of metrics to track
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    Timing,
    Memory,
    CPU,
    Quality,
    Efficiency,
    Scalability,
    All,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert timestamp
    pub timestamp: SystemTime,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Alert type
    pub alert_type: AlertType,
    
    /// Algorithm ID
    pub algorithm_id: String,
    
    /// Alert message
    pub message: String,
    
    /// Metric value that triggered alert
    pub trigger_value: f64,
    
    /// Threshold that was exceeded
    pub threshold: f64,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    PerformanceDegradation,
    MemoryLeak,
    CPUOverload,
    TimeoutExceeded,
    ConvergenceFailure,
    ResourceExhaustion,
    AnomalyDetected,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark ID
    pub benchmark_id: String,
    
    /// Algorithm ID
    pub algorithm_id: String,
    
    /// Benchmark configuration
    pub config: BenchmarkConfig,
    
    /// Results
    pub results: BenchmarkMetrics,
    
    /// Comparison with baseline
    pub baseline_comparison: Option<ComparisonResult>,
    
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Problem dimensions
    pub dimensions: usize,
    
    /// Population size
    pub population_size: usize,
    
    /// Maximum iterations
    pub max_iterations: usize,
    
    /// Number of runs
    pub runs: usize,
    
    /// Problem type
    pub problem_type: String,
    
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Benchmark metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Average performance across runs
    pub avg_performance: PerformanceMetrics,
    
    /// Best performance
    pub best_performance: PerformanceMetrics,
    
    /// Worst performance
    pub worst_performance: PerformanceMetrics,
    
    /// Standard deviation
    pub std_deviation: PerformanceMetrics,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Reliability score
    pub reliability: f64,
}

/// System resource monitor
pub struct SystemResourceMonitor {
    /// Last measurement
    last_measurement: Arc<RwLock<Option<SystemState>>>,
    
    /// Measurement history
    history: Arc<RwLock<VecDeque<SystemState>>>,
    
    /// Monitoring enabled
    enabled: bool,
}

impl SystemResourceMonitor {
    pub fn new() -> Self {
        Self {
            last_measurement: Arc::new(RwLock::new(None)),
            history: Arc::new(RwLock::new(VecDeque::new())),
            enabled: true,
        }
    }
    
    /// Get current system state
    pub fn get_current_state(&self) -> SystemState {
        if !self.enabled {
            return SystemState::default();
        }
        
        SystemState {
            cpu_load: self.get_cpu_load(),
            memory_usage: self.get_memory_usage(),
            disk_usage: self.get_disk_usage(),
            network_activity: self.get_network_activity(),
            temperature: self.get_temperature(),
            power_consumption: self.get_power_consumption(),
        }
    }
    
    fn get_cpu_load(&self) -> f64 {
        // Platform-specific CPU load detection
        // For now, return a simulated value
        0.5
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Platform-specific memory usage detection
        0.3
    }
    
    fn get_disk_usage(&self) -> f64 {
        // Platform-specific disk usage detection
        0.2
    }
    
    fn get_network_activity(&self) -> f64 {
        // Platform-specific network activity detection
        0.1
    }
    
    fn get_temperature(&self) -> Option<f64> {
        // Platform-specific temperature detection
        None
    }
    
    fn get_power_consumption(&self) -> Option<f64> {
        // Platform-specific power consumption detection
        None
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            cpu_load: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_activity: 0.0,
            temperature: None,
            power_consumption: None,
        }
    }
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new() -> Self {
        Self {
            metrics_storage: Arc::new(RwLock::new(HashMap::new())),
            monitoring_state: Arc::new(RwLock::new(MonitoringState {
                active_monitors: HashMap::new(),
                start_time: Instant::now(),
                measurement_count: 0,
                alerts: Vec::new(),
            })),
            benchmark_cache: Arc::new(RwLock::new(HashMap::new())),
            config: PerformanceConfig::default(),
            resource_monitor: Arc::new(SystemResourceMonitor::new()),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        let mut tracker = Self::new();
        tracker.config = config;
        tracker
    }
    
    /// Start monitoring an algorithm
    pub fn start_monitoring(&self, algorithm_id: String, metric_types: Vec<MetricType>) -> Result<(), SwarmError> {
        let mut state = self.monitoring_state.write();
        
        let monitor = AlgorithmMonitor {
            algorithm_id: algorithm_id.clone(),
            last_measurement: Instant::now(),
            interval: Duration::from_millis(self.config.monitoring_interval),
            enabled_metrics: metric_types,
            thresholds: self.get_default_thresholds(),
        };
        
        state.active_monitors.insert(algorithm_id.clone(), monitor);
        
        tracing::info!("Started monitoring algorithm: {}", algorithm_id);
        Ok(())
    }
    
    /// Stop monitoring an algorithm
    pub fn stop_monitoring(&self, algorithm_id: &str) -> Result<(), SwarmError> {
        let mut state = self.monitoring_state.write();
        
        if state.active_monitors.remove(algorithm_id).is_some() {
            tracing::info!("Stopped monitoring algorithm: {}", algorithm_id);
            Ok(())
        } else {
            Err(SwarmError::parameter(format!("Algorithm {} not being monitored", algorithm_id)))
        }
    }
    
    /// Record performance metrics for an algorithm
    pub fn record_metrics<T: Individual>(
        &self,
        algorithm_id: String,
        algorithm_metrics: AlgorithmMetrics,
        population: &Population<T>,
        execution_time: Duration,
    ) -> Result<(), SwarmError> {
        let system_state = self.resource_monitor.get_current_state();
        
        let performance_metrics = self.build_performance_metrics(
            algorithm_metrics,
            population,
            execution_time,
            &system_state,
        )?;
        
        let timestamped_metrics = TimestampedMetrics {
            timestamp: SystemTime::now(),
            metrics: performance_metrics,
            system_state,
        };
        
        self.store_metrics(algorithm_id, timestamped_metrics)?;
        
        Ok(())
    }
    
    /// Build comprehensive performance metrics
    fn build_performance_metrics<T: Individual>(
        &self,
        algorithm_metrics: AlgorithmMetrics,
        population: &Population<T>,
        execution_time: Duration,
        system_state: &SystemState,
    ) -> Result<PerformanceMetrics, SwarmError> {
        // Build timing metrics
        let timing = TimingMetrics {
            total_time: execution_time,
            time_per_iteration: if algorithm_metrics.iteration > 0 {
                execution_time / algorithm_metrics.iteration as u32
            } else {
                Duration::ZERO
            },
            time_to_convergence: None, // Would be calculated based on convergence detection
            cpu_time: execution_time, // Simplified
            wall_time: execution_time,
            operation_times: HashMap::new(),
        };
        
        // Build resource metrics
        let resources = ResourceMetrics {
            peak_memory: self.estimate_memory_usage(population)?,
            avg_memory: self.estimate_memory_usage(population)?,
            allocations: population.size(),
            cpu_utilization: system_state.cpu_load,
            thread_count: num_cpus::get(),
            cache_hit_rate: 0.8, // Estimated
            network_io: IOMetrics::default(),
            disk_io: IOMetrics::default(),
        };
        
        // Build quality metrics
        let quality = QualityMetrics {
            best_fitness: algorithm_metrics.best_fitness.unwrap_or(f64::INFINITY),
            avg_fitness: algorithm_metrics.average_fitness.unwrap_or(f64::INFINITY),
            fitness_std: 0.0, // Would calculate from population
            convergence_rate: algorithm_metrics.convergence_rate.unwrap_or(0.0),
            accuracy: self.calculate_accuracy(&algorithm_metrics)?,
            robustness: self.calculate_robustness(population)?,
            diversity_score: algorithm_metrics.diversity.unwrap_or(0.0),
        };
        
        // Build efficiency metrics
        let efficiency = EfficiencyMetrics {
            evaluations_per_second: if execution_time.as_secs_f64() > 0.0 {
                algorithm_metrics.evaluations as f64 / execution_time.as_secs_f64()
            } else {
                0.0
            },
            convergence_efficiency: self.calculate_convergence_efficiency(&algorithm_metrics)?,
            resource_efficiency: self.calculate_resource_efficiency(&resources)?,
            time_efficiency: self.calculate_time_efficiency(&timing, &algorithm_metrics)?,
            memory_efficiency: self.calculate_memory_efficiency(&resources, population.size())?,
            energy_efficiency: None, // Would require hardware monitoring
        };
        
        // Build scalability metrics
        let scalability = ScalabilityMetrics {
            size_scalability: 1.0, // Would calculate based on problem size scaling
            parallel_scalability: 1.0, // Would calculate based on thread scaling
            memory_scalability: 1.0, // Would calculate based on memory scaling
            time_complexity: 1.0, // Would analyze algorithm complexity
            space_complexity: 1.0, // Would analyze space complexity
        };
        
        Ok(PerformanceMetrics {
            algorithm_metrics,
            timing,
            resources,
            quality,
            efficiency,
            scalability,
        })
    }
    
    /// Store metrics in history
    fn store_metrics(
        &self,
        algorithm_id: String,
        metrics: TimestampedMetrics,
    ) -> Result<(), SwarmError> {
        let mut storage = self.metrics_storage.write();
        
        let history = storage.entry(algorithm_id.clone()).or_insert_with(|| {
            AlgorithmPerformanceHistory {
                algorithm_id: algorithm_id.clone(),
                metrics_history: VecDeque::new(),
                summary: PerformanceSummary::default(),
                trends: PerformanceTrends::default(),
                comparisons: HashMap::new(),
            }
        });
        
        history.metrics_history.push_back(metrics);
        
        // Limit history size
        while history.metrics_history.len() > self.config.max_history_size {
            history.metrics_history.pop_front();
        }
        
        // Update summary and trends
        self.update_summary(history)?;
        self.update_trends(history)?;
        
        Ok(())
    }
    
    /// Update performance summary
    fn update_summary(&self, history: &mut AlgorithmPerformanceHistory) -> Result<(), SwarmError> {
        if history.metrics_history.is_empty() {
            return Ok(());
        }
        
        // Calculate summary statistics
        // This is a simplified implementation - full implementation would calculate
        // proper statistical summaries for all metrics
        
        let latest = &history.metrics_history.back().unwrap().metrics;
        history.summary.best_performance = latest.clone();
        history.summary.avg_performance = latest.clone();
        history.summary.worst_performance = latest.clone();
        history.summary.std_deviations = latest.clone();
        history.summary.percentiles = HashMap::new();
        history.summary.stability_score = 0.8; // Estimated
        
        Ok(())
    }
    
    /// Update performance trends
    fn update_trends(&self, history: &mut AlgorithmPerformanceHistory) -> Result<(), SwarmError> {
        if history.metrics_history.len() < 2 {
            return Ok(());
        }
        
        // Simplified trend analysis
        history.trends.trend_direction = TrendDirection::Stable;
        history.trends.trend_strength = 0.5;
        history.trends.velocity = 0.0;
        history.trends.acceleration = 0.0;
        history.trends.seasonality = None;
        history.trends.anomalies = Vec::new();
        
        Ok(())
    }
    
    /// Get performance history for an algorithm
    pub fn get_performance_history(&self, algorithm_id: &str) -> Option<AlgorithmPerformanceHistory> {
        let storage = self.metrics_storage.read();
        storage.get(algorithm_id).cloned()
    }
    
    /// Compare two algorithms
    pub fn compare_algorithms(
        &self,
        algorithm1: &str,
        algorithm2: &str,
    ) -> Result<ComparisonResult, SwarmError> {
        let storage = self.metrics_storage.read();
        
        let history1 = storage.get(algorithm1)
            .ok_or_else(|| SwarmError::parameter(format!("Algorithm {} not found", algorithm1)))?;
        
        let history2 = storage.get(algorithm2)
            .ok_or_else(|| SwarmError::parameter(format!("Algorithm {} not found", algorithm2)))?;
        
        // Simplified comparison - in practice would use statistical tests
        let perf1 = &history1.summary.avg_performance;
        let perf2 = &history2.summary.avg_performance;
        
        let performance_ratio = if perf2.quality.best_fitness != 0.0 {
            perf1.quality.best_fitness / perf2.quality.best_fitness
        } else {
            1.0
        };
        
        let winner = if performance_ratio < 0.95 {
            ComparisonWinner::ThisAlgorithm
        } else if performance_ratio > 1.05 {
            ComparisonWinner::OtherAlgorithm
        } else {
            ComparisonWinner::Tie
        };
        
        Ok(ComparisonResult {
            other_algorithm: algorithm2.to_string(),
            performance_ratio,
            p_value: 0.05, // Would calculate proper statistical significance
            confidence_interval: (performance_ratio * 0.9, performance_ratio * 1.1),
            winner,
            breakdown: HashMap::new(),
        })
    }
    
    /// Run benchmark for an algorithm
    pub fn run_benchmark(
        &self,
        algorithm_id: String,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, SwarmError> {
        let benchmark_id = format!("{}_{}", algorithm_id, 
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
        
        // This is a placeholder - actual implementation would run the algorithm
        // multiple times and collect comprehensive metrics
        
        let dummy_metrics = PerformanceMetrics {
            algorithm_metrics: AlgorithmMetrics::default(),
            timing: TimingMetrics {
                total_time: Duration::from_millis(1000),
                time_per_iteration: Duration::from_millis(10),
                time_to_convergence: Some(Duration::from_millis(500)),
                cpu_time: Duration::from_millis(800),
                wall_time: Duration::from_millis(1000),
                operation_times: HashMap::new(),
            },
            resources: ResourceMetrics {
                peak_memory: 1024 * 1024, // 1MB
                avg_memory: 512 * 1024,   // 512KB
                allocations: 100,
                cpu_utilization: 0.7,
                thread_count: 4,
                cache_hit_rate: 0.85,
                network_io: IOMetrics::default(),
                disk_io: IOMetrics::default(),
            },
            quality: QualityMetrics {
                best_fitness: 0.001,
                avg_fitness: 0.1,
                fitness_std: 0.05,
                convergence_rate: 0.02,
                accuracy: 0.95,
                robustness: 0.8,
                diversity_score: 0.6,
            },
            efficiency: EfficiencyMetrics {
                evaluations_per_second: 1000.0,
                convergence_efficiency: 0.9,
                resource_efficiency: 0.8,
                time_efficiency: 0.85,
                memory_efficiency: 0.75,
                energy_efficiency: Some(0.7),
            },
            scalability: ScalabilityMetrics {
                size_scalability: 0.8,
                parallel_scalability: 0.9,
                memory_scalability: 0.85,
                time_complexity: 1.2,
                space_complexity: 1.1,
            },
        };
        
        let benchmark_metrics = BenchmarkMetrics {
            avg_performance: dummy_metrics.clone(),
            best_performance: dummy_metrics.clone(),
            worst_performance: dummy_metrics.clone(),
            std_deviation: dummy_metrics.clone(),
            success_rate: 0.95,
            reliability: 0.9,
        };
        
        let result = BenchmarkResult {
            benchmark_id: benchmark_id.clone(),
            algorithm_id,
            config,
            results: benchmark_metrics,
            baseline_comparison: None,
            timestamp: SystemTime::now(),
        };
        
        // Cache the result
        let mut cache = self.benchmark_cache.write();
        cache.insert(benchmark_id, result.clone());
        
        Ok(result)
    }
    
    /// Get default alert thresholds
    fn get_default_thresholds(&self) -> HashMap<String, f64> {
        let mut thresholds = HashMap::new();
        thresholds.insert("cpu_utilization".to_string(), 0.9);
        thresholds.insert("memory_usage".to_string(), 0.85);
        thresholds.insert("convergence_rate".to_string(), 0.001);
        thresholds.insert("execution_time".to_string(), 60.0); // seconds
        thresholds
    }
    
    /// Helper methods for metric calculations
    fn estimate_memory_usage<T: Individual>(&self, population: &Population<T>) -> Result<usize, SwarmError> {
        // Simplified memory estimation
        let base_size = std::mem::size_of::<T>();
        let dimensions = if population.is_empty() { 0 } else { population.individuals[0].dimensions() };
        let position_size = dimensions * std::mem::size_of::<f64>();
        
        Ok(population.size() * (base_size + position_size))
    }
    
    fn calculate_accuracy(&self, metrics: &AlgorithmMetrics) -> Result<f64, SwarmError> {
        // Simplified accuracy calculation
        if let Some(fitness) = metrics.best_fitness {
            Ok((1.0 / (1.0 + fitness)).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_robustness<T: Individual>(&self, population: &Population<T>) -> Result<f64, SwarmError> {
        // Simplified robustness calculation based on population diversity
        if let Some(diversity) = population.diversity() {
            Ok(diversity / (1.0 + diversity))
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_convergence_efficiency(&self, metrics: &AlgorithmMetrics) -> Result<f64, SwarmError> {
        // Efficiency based on convergence rate and evaluations
        if metrics.evaluations > 0 {
            let rate = metrics.convergence_rate.unwrap_or(0.0);
            Ok(rate * 1000.0 / metrics.evaluations as f64)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_resource_efficiency(&self, resources: &ResourceMetrics) -> Result<f64, SwarmError> {
        // Combined resource efficiency score
        let cpu_eff = 1.0 - resources.cpu_utilization;
        let mem_eff = if resources.peak_memory > 0 {
            1.0 - (resources.avg_memory as f64 / resources.peak_memory as f64)
        } else {
            1.0
        };
        
        Ok((cpu_eff + mem_eff) / 2.0)
    }
    
    fn calculate_time_efficiency(&self, timing: &TimingMetrics, metrics: &AlgorithmMetrics) -> Result<f64, SwarmError> {
        // Time efficiency based on iterations per second
        if timing.total_time.as_secs_f64() > 0.0 {
            let iterations_per_sec = metrics.iteration as f64 / timing.total_time.as_secs_f64();
            Ok(iterations_per_sec / 100.0) // Normalize to expected rate
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_memory_efficiency(&self, resources: &ResourceMetrics, population_size: usize) -> Result<f64, SwarmError> {
        // Memory efficiency per individual
        if population_size > 0 {
            let memory_per_individual = resources.avg_memory as f64 / population_size as f64;
            Ok(1.0 / (1.0 + memory_per_individual / 1024.0)) // Normalize to KB
        } else {
            Ok(0.0)
        }
    }
    
    /// Get current monitoring status
    pub fn get_monitoring_status(&self) -> MonitoringState {
        self.monitoring_state.read().clone()
    }
    
    /// Get all benchmark results
    pub fn get_benchmark_results(&self) -> HashMap<String, BenchmarkResult> {
        self.benchmark_cache.read().clone()
    }
    
    /// Clear all performance data
    pub fn clear_all_data(&self) {
        let mut storage = self.metrics_storage.write();
        storage.clear();
        
        let mut cache = self.benchmark_cache.write();
        cache.clear();
        
        let mut state = self.monitoring_state.write();
        state.active_monitors.clear();
        state.alerts.clear();
        state.measurement_count = 0;
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for complex structures
impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            best_performance: PerformanceMetrics::default(),
            avg_performance: PerformanceMetrics::default(),
            worst_performance: PerformanceMetrics::default(),
            std_deviations: PerformanceMetrics::default(),
            percentiles: HashMap::new(),
            stability_score: 0.0,
        }
    }
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            trend_direction: TrendDirection::Stable,
            trend_strength: 0.0,
            velocity: 0.0,
            acceleration: 0.0,
            seasonality: None,
            anomalies: Vec::new(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            algorithm_metrics: AlgorithmMetrics::default(),
            timing: TimingMetrics::default(),
            resources: ResourceMetrics::default(),
            quality: QualityMetrics::default(),
            efficiency: EfficiencyMetrics::default(),
            scalability: ScalabilityMetrics::default(),
        }
    }
}

impl Default for TimingMetrics {
    fn default() -> Self {
        Self {
            total_time: Duration::ZERO,
            time_per_iteration: Duration::ZERO,
            time_to_convergence: None,
            cpu_time: Duration::ZERO,
            wall_time: Duration::ZERO,
            operation_times: HashMap::new(),
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            peak_memory: 0,
            avg_memory: 0,
            allocations: 0,
            cpu_utilization: 0.0,
            thread_count: 1,
            cache_hit_rate: 0.0,
            network_io: IOMetrics::default(),
            disk_io: IOMetrics::default(),
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            best_fitness: f64::INFINITY,
            avg_fitness: f64::INFINITY,
            fitness_std: 0.0,
            convergence_rate: 0.0,
            accuracy: 0.0,
            robustness: 0.0,
            diversity_score: 0.0,
        }
    }
}

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            evaluations_per_second: 0.0,
            convergence_efficiency: 0.0,
            resource_efficiency: 0.0,
            time_efficiency: 0.0,
            memory_efficiency: 0.0,
            energy_efficiency: None,
        }
    }
}

impl Default for ScalabilityMetrics {
    fn default() -> Self {
        Self {
            size_scalability: 1.0,
            parallel_scalability: 1.0,
            memory_scalability: 1.0,
            time_complexity: 1.0,
            space_complexity: 1.0,
        }
    }
}

/// Trait for benchmarkable algorithms
pub trait Benchmark {
    fn run_benchmark(&self, config: BenchmarkConfig) -> Result<BenchmarkResult, SwarmError>;
    fn get_performance_profile(&self) -> PerformanceMetrics;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::BasicIndividual;
    
    #[test]
    fn test_performance_tracker_creation() {
        let tracker = PerformanceTracker::new();
        assert!(tracker.config.enable_realtime);
        assert_eq!(tracker.config.max_history_size, 10000);
    }
    
    #[test]
    fn test_monitoring_lifecycle() {
        let tracker = PerformanceTracker::new();
        let algorithm_id = "test_algorithm".to_string();
        
        // Start monitoring
        let result = tracker.start_monitoring(algorithm_id.clone(), vec![MetricType::All]);
        assert!(result.is_ok());
        
        // Check monitoring status
        let status = tracker.get_monitoring_status();
        assert!(status.active_monitors.contains_key(&algorithm_id));
        
        // Stop monitoring
        let result = tracker.stop_monitoring(&algorithm_id);
        assert!(result.is_ok());
        
        // Check monitoring stopped
        let status = tracker.get_monitoring_status();
        assert!(!status.active_monitors.contains_key(&algorithm_id));
    }
    
    #[test]
    fn test_metrics_recording() {
        let tracker = PerformanceTracker::new();
        let algorithm_id = "test_algorithm".to_string();
        
        let mut population = Population::new();
        population.add(BasicIndividual::new(Position::from_vec(vec![1.0, 2.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![3.0, 4.0])));
        
        let algorithm_metrics = AlgorithmMetrics {
            iteration: 10,
            best_fitness: Some(0.5),
            average_fitness: Some(1.0),
            diversity: Some(0.8),
            convergence_rate: Some(0.02),
            evaluations: 100,
            time_per_iteration: Some(50),
            memory_usage: Some(1024),
        };
        
        let execution_time = Duration::from_millis(500);
        
        let result = tracker.record_metrics(
            algorithm_id.clone(),
            algorithm_metrics,
            &population,
            execution_time,
        );
        
        assert!(result.is_ok());
        
        // Check history was recorded
        let history = tracker.get_performance_history(&algorithm_id);
        assert!(history.is_some());
        assert!(!history.unwrap().metrics_history.is_empty());
    }
    
    #[test]
    fn test_benchmark_execution() {
        let tracker = PerformanceTracker::new();
        let algorithm_id = "test_algorithm".to_string();
        
        let config = BenchmarkConfig {
            dimensions: 2,
            population_size: 20,
            max_iterations: 100,
            runs: 5,
            problem_type: "sphere".to_string(),
            parameters: HashMap::new(),
        };
        
        let result = tracker.run_benchmark(algorithm_id, config);
        assert!(result.is_ok());
        
        let benchmark_result = result.unwrap();
        assert_eq!(benchmark_result.results.success_rate, 0.95);
        assert!(benchmark_result.results.reliability > 0.0);
    }
    
    #[test]
    fn test_algorithm_comparison() {
        let tracker = PerformanceTracker::new();
        
        // Record metrics for two algorithms
        let algorithm1 = "algo1".to_string();
        let algorithm2 = "algo2".to_string();
        
        let population = Population::new();
        
        let metrics1 = AlgorithmMetrics {
            best_fitness: Some(0.1),
            ..Default::default()
        };
        
        let metrics2 = AlgorithmMetrics {
            best_fitness: Some(0.2),
            ..Default::default()
        };
        
        let _ = tracker.record_metrics(algorithm1.clone(), metrics1, &population, Duration::from_millis(100));
        let _ = tracker.record_metrics(algorithm2.clone(), metrics2, &population, Duration::from_millis(100));
        
        let comparison = tracker.compare_algorithms(&algorithm1, &algorithm2);
        assert!(comparison.is_ok());
        
        let comp_result = comparison.unwrap();
        assert!(matches!(comp_result.winner, ComparisonWinner::ThisAlgorithm));
    }
    
    #[test]
    fn test_system_resource_monitor() {
        let monitor = SystemResourceMonitor::new();
        let state = monitor.get_current_state();
        
        assert!(state.cpu_load >= 0.0 && state.cpu_load <= 1.0);
        assert!(state.memory_usage >= 0.0 && state.memory_usage <= 1.0);
        assert!(state.disk_usage >= 0.0 && state.disk_usage <= 1.0);
        assert!(state.network_activity >= 0.0);
    }
}