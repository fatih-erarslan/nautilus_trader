//! Core benchmarking infrastructure
//!
//! Fundamental types and traits for the benchmarking system.

use crate::core::{SwarmAlgorithm, SwarmError, OptimizationProblem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use std::sync::Arc;

/// Comprehensive benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Test name and description
    pub name: String,
    pub description: String,
    
    /// Algorithm configuration
    pub algorithm_name: String,
    pub algorithm_params: HashMap<String, serde_json::Value>,
    
    /// Problem configuration
    pub problem_name: String,
    pub problem_dimensions: usize,
    pub bounds: (f64, f64),
    
    /// Execution parameters
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub population_size: usize,
    pub independent_runs: usize,
    
    /// Performance criteria
    pub target_fitness: Option<f64>,
    pub convergence_threshold: f64,
    pub timeout: Duration,
    
    /// Resource monitoring
    pub monitor_memory: bool,
    pub monitor_cpu: bool,
    pub monitor_convergence: bool,
    pub collect_detailed_metrics: bool,
    
    /// Statistical requirements
    pub significance_level: f64,
    pub confidence_level: f64,
    pub min_sample_size: usize,
    
    /// Enterprise features
    pub export_prometheus: bool,
    pub generate_report: bool,
    pub store_baseline: bool,
    pub check_regression: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "Default Benchmark".to_string(),
            description: "Standard performance benchmark".to_string(),
            algorithm_name: "PSO".to_string(),
            algorithm_params: HashMap::new(),
            problem_name: "Sphere".to_string(),
            problem_dimensions: 30,
            bounds: (-100.0, 100.0),
            max_iterations: 1000,
            max_evaluations: 30000,
            population_size: 50,
            independent_runs: 30,
            target_fitness: Some(1e-6),
            convergence_threshold: 1e-8,
            timeout: Duration::from_secs(300),
            monitor_memory: true,
            monitor_cpu: true,
            monitor_convergence: true,
            collect_detailed_metrics: true,
            significance_level: 0.05,
            confidence_level: 0.95,
            min_sample_size: 25,
            export_prometheus: true,
            generate_report: true,
            store_baseline: false,
            check_regression: true,
        }
    }
}

/// Comprehensive benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Configuration used
    pub config: BenchmarkConfig,
    
    /// Execution metadata
    pub execution_id: String,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub total_duration: Duration,
    pub status: BenchmarkStatus,
    
    /// Statistical results
    pub statistics: BenchmarkStatistics,
    
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    
    /// Resource utilization
    pub resource_utilization: ResourceMetrics,
    
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    
    /// Individual run results
    pub individual_runs: Vec<RunResult>,
    
    /// Comparative analysis (if applicable)
    pub comparative_analysis: Option<ComparativeAnalysis>,
    
    /// Enterprise metrics
    pub enterprise_metrics: EnterpriseMetrics,
    
    /// Error information (if any)
    pub errors: Vec<BenchmarkError>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Benchmark execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BenchmarkStatus {
    /// Successfully completed
    Completed,
    
    /// Completed with warnings
    CompletedWithWarnings,
    
    /// Partially completed (some runs failed)
    PartiallyCompleted,
    
    /// Failed due to error
    Failed,
    
    /// Timed out
    TimedOut,
    
    /// Cancelled by user
    Cancelled,
    
    /// Currently running
    Running,
    
    /// Queued for execution
    Queued,
}

/// Statistical analysis of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    /// Fitness statistics
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub mean_fitness: f64,
    pub median_fitness: f64,
    pub std_deviation: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    
    /// Success metrics
    pub success_rate: f64,
    pub convergence_rate: f64,
    pub target_achievement_rate: f64,
    
    /// Confidence intervals
    pub confidence_interval_95: (f64, f64),
    pub confidence_interval_99: (f64, f64),
    
    /// Distribution analysis
    pub percentiles: HashMap<u8, f64>, // 25th, 50th, 75th, 90th, 95th, 99th
    pub quartiles: (f64, f64, f64), // Q1, Q2, Q3
    pub interquartile_range: f64,
    
    /// Outlier detection
    pub outliers: Vec<usize>, // Indices of outlier runs
    pub outlier_threshold: f64,
    
    /// Normality tests
    pub shapiro_wilk_p_value: Option<f64>,
    pub anderson_darling_statistic: Option<f64>,
    pub is_normally_distributed: bool,
}

/// Performance metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Execution timing
    pub mean_execution_time: Duration,
    pub std_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    
    /// Algorithm efficiency
    pub evaluations_per_second: f64,
    pub iterations_per_second: f64,
    pub convergence_speed: f64,
    pub efficiency_score: f64,
    
    /// Memory performance
    pub peak_memory_usage: usize,
    pub average_memory_usage: usize,
    pub memory_efficiency: f64,
    
    /// CPU utilization
    pub cpu_utilization_mean: f64,
    pub cpu_utilization_peak: f64,
    pub cpu_efficiency: f64,
    
    /// Scalability metrics
    pub scalability_factor: f64,
    pub parallelization_efficiency: f64,
    pub thread_utilization: f64,
}

/// System resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// System resources
    pub cpu_cores_used: usize,
    pub memory_total: usize,
    pub memory_available: usize,
    pub swap_usage: usize,
    
    /// Process resources
    pub process_cpu_time: Duration,
    pub process_memory_peak: usize,
    pub process_memory_average: usize,
    pub thread_count_peak: usize,
    pub thread_count_average: usize,
    
    /// IO metrics
    pub disk_reads: u64,
    pub disk_writes: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    
    /// Cache performance
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
}

/// Convergence behavior analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    /// Convergence timing
    pub convergence_iteration: Option<usize>,
    pub convergence_evaluation: Option<usize>,
    pub convergence_time: Option<Duration>,
    
    /// Convergence quality
    pub convergence_rate: f64,
    pub convergence_consistency: f64,
    pub convergence_stability: f64,
    
    /// Progress tracking
    pub improvement_phases: Vec<ImprovementPhase>,
    pub stagnation_periods: Vec<StagnationPeriod>,
    pub diversity_evolution: Vec<f64>,
    
    /// Convergence patterns
    pub convergence_pattern: ConvergencePattern,
    pub exploration_exploitation_balance: f64,
    pub premature_convergence_risk: f64,
}

/// Algorithm improvement phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementPhase {
    pub start_iteration: usize,
    pub end_iteration: usize,
    pub improvement_rate: f64,
    pub phase_type: ImprovementType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementType {
    Rapid,
    Steady,
    Slow,
    Sporadic,
}

/// Stagnation period identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagnationPeriod {
    pub start_iteration: usize,
    pub end_iteration: usize,
    pub duration: usize,
    pub severity: StagnationSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StagnationSeverity {
    Mild,
    Moderate,
    Severe,
    Critical,
}

/// Convergence pattern classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergencePattern {
    Exponential,
    Linear,
    Logarithmic,
    Stepwise,
    Oscillating,
    Irregular,
    PrematureConvergence,
    NoConvergence,
}

/// Individual run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub run_id: usize,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub duration: Duration,
    pub status: RunStatus,
    pub final_fitness: f64,
    pub target_achieved: bool,
    pub iterations_performed: usize,
    pub evaluations_performed: usize,
    pub convergence_iteration: Option<usize>,
    pub peak_memory: usize,
    pub average_cpu: f64,
    pub fitness_history: Vec<f64>,
    pub diversity_history: Vec<f64>,
    pub error: Option<String>,
}

/// Status of individual run
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RunStatus {
    Success,
    TargetAchieved,
    MaxIterationsReached,
    MaxEvaluationsReached,
    TimeoutReached,
    Error(String),
}

/// Comparative analysis between algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub baseline_algorithm: String,
    pub comparison_algorithms: Vec<String>,
    pub statistical_tests: Vec<StatisticalTest>,
    pub effect_sizes: HashMap<String, f64>,
    pub rankings: Vec<AlgorithmRanking>,
    pub recommendations: Vec<String>,
}

/// Statistical significance test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_significant: bool,
    pub effect_size: f64,
    pub confidence_interval: (f64, f64),
}

/// Algorithm performance ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRanking {
    pub rank: usize,
    pub algorithm_name: String,
    pub score: f64,
    pub criteria: String,
}

/// Enterprise-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseMetrics {
    pub cost_analysis: CostAnalysis,
    pub sla_compliance: SlaCompliance,
    pub quality_metrics: QualityMetrics,
    pub security_metrics: SecurityMetrics,
    pub audit_trail: Vec<AuditEvent>,
}

/// Cost analysis for enterprise environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    pub compute_cost: f64,
    pub memory_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
    pub total_cost: f64,
    pub cost_per_evaluation: f64,
    pub cost_efficiency_score: f64,
}

/// Service Level Agreement compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaCompliance {
    pub performance_sla_met: bool,
    pub availability_sla_met: bool,
    pub latency_sla_met: bool,
    pub throughput_sla_met: bool,
    pub overall_compliance_score: f64,
}

/// Quality assurance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub result_accuracy: f64,
    pub result_consistency: f64,
    pub reproducibility_score: f64,
    pub validation_status: ValidationStatus,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Validated,
    PartiallyValidated,
    NotValidated,
    ValidationFailed,
}

/// Security-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub data_integrity_verified: bool,
    pub access_control_compliance: bool,
    pub encryption_used: bool,
    pub audit_logging_enabled: bool,
    pub security_score: f64,
}

/// Audit event for compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: SystemTime,
    pub event_type: String,
    pub description: String,
    pub user_id: Option<String>,
    pub data_hash: String,
}

/// Benchmark suite containing multiple configurations
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub name: String,
    pub description: String,
    pub configs: Vec<BenchmarkConfig>,
    pub execution_order: ExecutionOrder,
    pub parallel_execution: bool,
    pub max_concurrent: usize,
}

#[derive(Debug, Clone)]
pub enum ExecutionOrder {
    Sequential,
    Parallel,
    Priority,
    Random,
}

impl BenchmarkSuite {
    /// Create comprehensive benchmark suite for all algorithms
    pub fn comprehensive() -> Self {
        let mut configs = Vec::new();
        
        // Standard problems for all algorithms
        let problems = vec![
            ("Sphere", 30),
            ("Rosenbrock", 30), 
            ("Rastrigin", 30),
            ("Griewank", 30),
            ("Ackley", 30),
        ];
        
        let algorithms = crate::available_algorithms();
        
        for algorithm in algorithms {
            for (problem, dimensions) in &problems {
                configs.push(BenchmarkConfig {
                    name: format!("{}_on_{}", algorithm, problem),
                    description: format!("Comprehensive benchmark of {} on {} function", algorithm, problem),
                    algorithm_name: algorithm.to_string(),
                    problem_name: problem.to_string(),
                    problem_dimensions: *dimensions,
                    independent_runs: 30,
                    max_iterations: 1000,
                    max_evaluations: 30000,
                    ..Default::default()
                });
            }
        }
        
        Self {
            name: "Comprehensive Algorithm Benchmark".to_string(),
            description: "Complete performance evaluation of all swarm algorithms".to_string(),
            configs,
            execution_order: ExecutionOrder::Parallel,
            parallel_execution: true,
            max_concurrent: num_cpus::get(),
        }
    }
    
    /// Create quick performance check suite for CI/CD
    pub fn quick_check() -> Self {
        let mut configs = Vec::new();
        
        // Limited set for quick validation
        let algorithms = vec!["PSO", "GWO", "WOA"];
        let problems = vec![("Sphere", 10), ("Rastrigin", 10)];
        
        for algorithm in algorithms {
            for (problem, dimensions) in &problems {
                configs.push(BenchmarkConfig {
                    name: format!("Quick_{}_on_{}", algorithm, problem),
                    algorithm_name: algorithm.to_string(),
                    problem_name: problem.to_string(),
                    problem_dimensions: dimensions,
                    independent_runs: 5,
                    max_iterations: 100,
                    max_evaluations: 5000,
                    timeout: Duration::from_secs(60),
                    collect_detailed_metrics: false,
                    ..Default::default()
                });
            }
        }
        
        Self {
            name: "Quick Performance Check".to_string(),
            description: "Fast validation benchmark for CI/CD".to_string(),
            configs,
            execution_order: ExecutionOrder::Parallel,
            parallel_execution: true,
            max_concurrent: 4,
        }
    }
}

/// Main benchmark runner
pub struct BenchmarkRunner {
    executor: Arc<RwLock<BenchmarkExecutor>>,
    monitor: Arc<RwLock<PerformanceMonitor>>,
    analyzer: Arc<RwLock<StatisticalAnalyzer>>,
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self {
            executor: Arc::new(RwLock::new(BenchmarkExecutor::new())),
            monitor: Arc::new(RwLock::new(PerformanceMonitor::new())),
            analyzer: Arc::new(RwLock::new(StatisticalAnalyzer::new())),
        }
    }
    
    /// Run a complete benchmark suite
    pub async fn run_suite(&self, suite: BenchmarkSuite) -> Result<Vec<BenchmarkResult>, BenchmarkError> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        
        tracing::info!("Starting benchmark suite: {}", suite.name);
        
        if suite.parallel_execution {
            results = self.run_parallel(&suite).await?;
        } else {
            results = self.run_sequential(&suite).await?;
        }
        
        let total_duration = start_time.elapsed();
        tracing::info!(
            "Benchmark suite completed in {:?}. {} benchmarks executed.",
            total_duration,
            results.len()
        );
        
        Ok(results)
    }
    
    async fn run_parallel(&self, suite: &BenchmarkSuite) -> Result<Vec<BenchmarkResult>, BenchmarkError> {
        use futures::stream::{self, StreamExt};
        
        let semaphore = Arc::new(tokio::sync::Semaphore::new(suite.max_concurrent));
        let results: Vec<_> = stream::iter(suite.configs.iter())
            .map(|config| {
                let semaphore = semaphore.clone();
                let executor = self.executor.clone();
                let monitor = self.monitor.clone();
                
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    self.run_single_benchmark(config, &executor, &monitor).await
                }
            })
            .buffer_unordered(suite.max_concurrent)
            .collect()
            .await;
        
        results.into_iter().collect()
    }
    
    async fn run_sequential(&self, suite: &BenchmarkSuite) -> Result<Vec<BenchmarkResult>, BenchmarkError> {
        let mut results = Vec::new();
        
        for config in &suite.configs {
            let result = self.run_single_benchmark(config, &self.executor, &self.monitor).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    async fn run_single_benchmark(
        &self,
        config: &BenchmarkConfig,
        executor: &Arc<RwLock<BenchmarkExecutor>>,
        monitor: &Arc<RwLock<PerformanceMonitor>>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        
        tracing::info!("Starting benchmark: {} (ID: {})", config.name, execution_id);
        
        let start_time = SystemTime::now();
        let mut individual_runs = Vec::new();
        
        // Start monitoring
        {
            let mut monitor_guard = monitor.write().await;
            monitor_guard.start_monitoring(&execution_id)?;
        }
        
        // Execute independent runs
        for run_id in 0..config.independent_runs {
            match self.execute_single_run(config, run_id, executor).await {
                Ok(run_result) => individual_runs.push(run_result),
                Err(e) => {
                    tracing::warn!("Run {} failed: {}", run_id, e);
                    individual_runs.push(RunResult {
                        run_id,
                        status: RunStatus::Error(e.to_string()),
                        start_time: SystemTime::now(),
                        end_time: SystemTime::now(),
                        duration: Duration::from_secs(0),
                        final_fitness: f64::INFINITY,
                        target_achieved: false,
                        iterations_performed: 0,
                        evaluations_performed: 0,
                        convergence_iteration: None,
                        peak_memory: 0,
                        average_cpu: 0.0,
                        fitness_history: vec![],
                        diversity_history: vec![],
                        error: Some(e.to_string()),
                    });
                }
            }
        }
        
        // Stop monitoring and collect metrics
        let (performance_metrics, resource_metrics) = {
            let mut monitor_guard = monitor.write().await;
            monitor_guard.stop_monitoring(&execution_id)?
        };
        
        let end_time = SystemTime::now();
        let total_duration = end_time.duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));
        
        // Analyze results
        let analyzer = self.analyzer.read().await;
        let statistics = analyzer.calculate_statistics(&individual_runs)?;
        let convergence_analysis = analyzer.analyze_convergence(&individual_runs)?;
        
        let result = BenchmarkResult {
            config: config.clone(),
            execution_id,
            start_time,
            end_time,
            total_duration,
            status: self.determine_status(&individual_runs),
            statistics,
            performance_metrics,
            resource_utilization: resource_metrics,
            convergence_analysis,
            individual_runs,
            comparative_analysis: None,
            enterprise_metrics: EnterpriseMetrics::default(),
            errors: Vec::new(),
            metadata: HashMap::new(),
        };
        
        tracing::info!("Benchmark completed: {}", config.name);
        Ok(result)
    }
    
    async fn execute_single_run(
        &self,
        config: &BenchmarkConfig,
        run_id: usize,
        executor: &Arc<RwLock<BenchmarkExecutor>>,
    ) -> Result<RunResult, BenchmarkError> {
        let executor_guard = executor.read().await;
        executor_guard.execute_run(config, run_id).await
    }
    
    fn determine_status(&self, runs: &[RunResult]) -> BenchmarkStatus {
        let total_runs = runs.len();
        let successful_runs = runs.iter().filter(|r| r.status == RunStatus::Success).count();
        let failed_runs = total_runs - successful_runs;
        
        if failed_runs == 0 {
            BenchmarkStatus::Completed
        } else if successful_runs > total_runs / 2 {
            BenchmarkStatus::CompletedWithWarnings
        } else if successful_runs > 0 {
            BenchmarkStatus::PartiallyCompleted
        } else {
            BenchmarkStatus::Failed
        }
    }
}

// Forward declarations for modules
use crate::benchmarks::{
    execution::BenchmarkExecutor,
    monitoring::PerformanceMonitor,
    analysis::StatisticalAnalyzer,
};

impl Default for EnterpriseMetrics {
    fn default() -> Self {
        Self {
            cost_analysis: CostAnalysis {
                compute_cost: 0.0,
                memory_cost: 0.0,
                storage_cost: 0.0,
                network_cost: 0.0,
                total_cost: 0.0,
                cost_per_evaluation: 0.0,
                cost_efficiency_score: 1.0,
            },
            sla_compliance: SlaCompliance {
                performance_sla_met: true,
                availability_sla_met: true,
                latency_sla_met: true,
                throughput_sla_met: true,
                overall_compliance_score: 1.0,
            },
            quality_metrics: QualityMetrics {
                result_accuracy: 1.0,
                result_consistency: 1.0,
                reproducibility_score: 1.0,
                validation_status: ValidationStatus::NotValidated,
                quality_score: 1.0,
            },
            security_metrics: SecurityMetrics {
                data_integrity_verified: true,
                access_control_compliance: true,
                encryption_used: false,
                audit_logging_enabled: true,
                security_score: 0.8,
            },
            audit_trail: Vec::new(),
        }
    }
}

/// Benchmark-specific error types
#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Algorithm error: {0}")]
    AlgorithmError(#[from] SwarmError),
    
    #[error("Timeout error")]
    TimeoutError,
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Initialize core benchmark infrastructure
pub async fn initialize_benchmark_core() -> Result<(), BenchmarkError> {
    tracing::info!("Initializing benchmark core infrastructure");
    Ok(())
}

/// Benchmark metrics for enterprise monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub total_benchmarks_run: u64,
    pub successful_benchmarks: u64,
    pub failed_benchmarks: u64,
    pub average_execution_time: Duration,
    pub total_cpu_hours: f64,
    pub total_memory_gb_hours: f64,
    pub benchmark_throughput: f64,
    pub quality_score: f64,
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self {
            total_benchmarks_run: 0,
            successful_benchmarks: 0,
            failed_benchmarks: 0,
            average_execution_time: Duration::from_secs(0),
            total_cpu_hours: 0.0,
            total_memory_gb_hours: 0.0,
            benchmark_throughput: 0.0,
            quality_score: 1.0,
        }
    }
}