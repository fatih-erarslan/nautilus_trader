//! Enterprise Performance Benchmarking System
//!
//! Comprehensive benchmarking infrastructure for swarm intelligence algorithms
//! with enterprise-grade monitoring, statistical analysis, and reporting.
//!
//! ## Features
//!
//! - **Comprehensive Algorithm Benchmarking**: All 13+ swarm algorithms
//! - **Standard Benchmark Problems**: CEC 2017, IEEE functions, custom problems  
//! - **Real-time Performance Monitoring**: CPU, memory, convergence tracking
//! - **Statistical Analysis**: Significance testing, confidence intervals
//! - **Enterprise Integration**: Prometheus metrics, Grafana dashboards
//! - **Performance Regression Detection**: Automated baseline comparison
//! - **Publication-Quality Reporting**: Scientific rigor with statistical validation

pub mod core;
pub mod framework;
pub mod problems;
pub mod monitoring;
pub mod analysis;
pub mod reporting;
pub mod enterprise;
pub mod execution;

// Core benchmarking infrastructure
pub use core::{
    BenchmarkSuite, BenchmarkRunner, BenchmarkConfig, BenchmarkResult,
    BenchmarkMetrics, BenchmarkStatus, BenchmarkError
};

// Benchmarking framework
pub use framework::{
    PerformanceBenchmark, AlgorithmBenchmark, ComparativeBenchmark,
    RegressionBenchmark, ScalabilityBenchmark, MemoryBenchmark
};

// Standard benchmark problems
pub use problems::{
    StandardBenchmarkProblems, CEC2017Problems, IEEEBenchmarkFunctions,
    CustomBenchmarkProblem, BenchmarkProblemSuite
};

// Monitoring and metrics
pub use monitoring::{
    PerformanceMonitor, RealTimeMetrics, SystemResourceMonitor,
    AlgorithmTracker, ConvergenceMonitor, ResourceUtilizationTracker
};

// Statistical analysis
pub use analysis::{
    StatisticalAnalyzer, PerformanceAnalysis, ComparisonAnalysis,
    RegressionAnalysis, SignificanceTests, ConfidenceIntervals
};

// Reporting system
pub use reporting::{
    BenchmarkReporter, PerformanceReport, ComparisonReport,
    TrendReport, ExecutiveSummary, ScientificReport
};

// Enterprise features
pub use enterprise::{
    PrometheusIntegration, GrafanaDashboard, AlertingSystem,
    PerformanceBaselines, ContinuousIntegration, AutomatedReporting
};

// Execution engine
pub use execution::{
    BenchmarkExecutor, ParallelExecutor, DistributedExecutor,
    ExecutionPlan, ExecutionStrategy, ResourceAllocation
};

/// Initialize the benchmarking system
pub async fn initialize_benchmarking() -> Result<(), BenchmarkError> {
    // Initialize core systems
    core::initialize_benchmark_core().await?;
    
    // Set up monitoring infrastructure
    monitoring::initialize_monitoring().await?;
    
    // Configure enterprise integrations
    enterprise::initialize_enterprise_features().await?;
    
    tracing::info!("Enterprise benchmarking system initialized successfully");
    Ok(())
}

/// Run comprehensive algorithm benchmarks
pub async fn run_comprehensive_benchmarks() -> Result<Vec<BenchmarkResult>, BenchmarkError> {
    let suite = BenchmarkSuite::comprehensive();
    let runner = BenchmarkRunner::new();
    
    runner.run_suite(suite).await
}

/// Quick performance check for CI/CD
pub async fn quick_performance_check() -> Result<BenchmarkResult, BenchmarkError> {
    let suite = BenchmarkSuite::quick_check();
    let runner = BenchmarkRunner::new();
    
    let results = runner.run_suite(suite).await?;
    Ok(results.into_iter().next().unwrap())
}

/// Get current performance baselines
pub fn get_performance_baselines() -> Vec<enterprise::PerformanceBaseline> {
    enterprise::get_current_baselines()
}

/// Export benchmarking results for external analysis
pub fn export_results(
    results: &[BenchmarkResult],
    format: reporting::ExportFormat,
) -> Result<String, BenchmarkError> {
    reporting::export_benchmark_results(results, format)
}