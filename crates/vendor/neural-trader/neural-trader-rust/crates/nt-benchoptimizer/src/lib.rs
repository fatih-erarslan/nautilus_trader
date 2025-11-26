//! # Neural Trader Benchmark Optimizer
//!
//! High-performance benchmarking and optimization toolkit for Neural Trader packages.
//!
//! ## Features
//! - Multi-threaded benchmarking with rayon
//! - Statistical analysis (mean, median, p95, p99)
//! - Memory profiling and leak detection
//! - Bundle size analysis
//! - Dependency validation
//! - Optimization recommendations
//! - SIMD-accelerated computations

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};

mod benchmarker;
mod validator;
mod optimizer;
mod metrics;
mod reporter;

pub use benchmarker::Benchmarker;
pub use validator::Validator;
pub use optimizer::Optimizer;
pub use metrics::{BenchmarkMetrics, ValidationResult, OptimizationSuggestion};

/// Initialize the benchmarker with configuration
#[napi]
pub fn initialize(config: Option<String>) -> Result<()> {
    // Parse optional config
    if let Some(cfg) = config {
        let _: serde_json::Value = serde_json::from_str(&cfg)
            .map_err(|e| Error::from_reason(format!("Invalid config: {}", e)))?;
    }

    // Initialize thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .map_err(|e| Error::from_reason(format!("Failed to initialize thread pool: {}", e)))?;

    Ok(())
}

/// Benchmark a single package's performance
#[napi]
pub async fn benchmark_package(package_path: String, options: Option<BenchmarkOptions>) -> Result<BenchmarkResult> {
    let benchmarker = Benchmarker::new(package_path)?;
    let opts = options.unwrap_or_default();

    benchmarker.run_benchmark(opts).await
}

/// Validate package structure and dependencies
#[napi]
pub async fn validate_package(package_path: String, options: Option<ValidationOptions>) -> Result<ValidationReport> {
    let validator = Validator::new(package_path)?;
    let opts = options.unwrap_or_default();

    validator.validate(opts).await
}

/// Analyze and suggest optimizations
#[napi]
pub async fn optimize_package(package_path: String, options: Option<OptimizationOptions>) -> Result<OptimizationReport> {
    let optimizer = Optimizer::new(package_path)?;
    let opts = options.unwrap_or_default();

    optimizer.analyze(opts).await
}

/// Benchmark all Neural Trader packages in parallel
#[napi]
pub async fn benchmark_all(workspace_path: String, options: Option<BenchmarkOptions>) -> Result<Vec<BenchmarkResult>> {
    let packages = discover_packages(&workspace_path)?;
    let opts = options.unwrap_or_default();

    // Use rayon for parallel benchmarking
    let results: Vec<_> = tokio::task::spawn_blocking(move || {
        packages.into_iter()
            .map(|pkg_path| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let benchmarker = Benchmarker::new(pkg_path)?;
                    benchmarker.run_benchmark(opts.clone()).await
                })
            })
            .collect::<Result<Vec<_>>>()
    })
    .await
    .map_err(|e| Error::from_reason(format!("Benchmark task failed: {}", e)))??;

    Ok(results)
}

/// Generate comprehensive performance report
#[napi]
pub async fn generate_report(
    benchmark_results: Vec<BenchmarkResult>,
    validation_reports: Vec<ValidationReport>,
    optimization_reports: Vec<OptimizationReport>,
    output_path: String,
    format: Option<String>,
) -> Result<String> {
    let fmt = format.as_deref().unwrap_or("json");

    let report = reporter::generate_comprehensive_report(
        benchmark_results,
        validation_reports,
        optimization_reports,
        fmt,
    )?;

    // Write to file
    tokio::fs::write(&output_path, &report)
        .await
        .map_err(|e| Error::from_reason(format!("Failed to write report: {}", e)))?;

    Ok(output_path)
}

/// Compare two benchmark results
#[napi]
pub fn compare_benchmarks(baseline: BenchmarkResult, current: BenchmarkResult) -> Result<ComparisonReport> {
    Ok(ComparisonReport {
        package_name: current.package_name.clone(),
        performance_delta: calculate_delta(baseline.execution_time_ms, current.execution_time_ms),
        memory_delta: calculate_delta(baseline.memory_usage_mb as f64, current.memory_usage_mb as f64),
        bundle_size_delta: calculate_delta(baseline.bundle_size_kb as f64, current.bundle_size_kb as f64),
        regression_detected: current.execution_time_ms > baseline.execution_time_ms * 1.1,
        improvements: vec![],
        regressions: vec![],
    })
}

/// Get system capabilities and configuration
#[napi]
pub fn get_system_info() -> Result<SystemInfo> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    Ok(SystemInfo {
        cpu_count: num_cpus::get() as i32,
        total_memory_gb: (sys.total_memory() as f64) / (1024.0 * 1024.0 * 1024.0),
        available_memory_gb: (sys.available_memory() as f64) / (1024.0 * 1024.0 * 1024.0),
        os_version: System::name().map(|s| s.to_string()).unwrap_or_else(|| "Unknown".to_string()),
        architecture: std::env::consts::ARCH.to_string(),
    })
}

// ============================================================================
// Type Definitions
// ============================================================================

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkOptions {
    pub iterations: Option<i32>,
    pub warmup_iterations: Option<i32>,
    pub include_memory_profiling: Option<bool>,
    pub include_bundle_analysis: Option<bool>,
    pub parallel: Option<bool>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationOptions {
    pub check_dependencies: Option<bool>,
    pub check_typescript: Option<bool>,
    pub check_napi_bindings: Option<bool>,
    pub strict_mode: Option<bool>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationOptions {
    pub analyze_bundle: Option<bool>,
    pub analyze_dependencies: Option<bool>,
    pub analyze_code_splitting: Option<bool>,
    pub suggest_refactoring: Option<bool>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub package_name: String,
    pub package_path: String,
    pub execution_time_ms: f64,
    pub memory_usage_mb: i64,
    pub bundle_size_kb: i64,
    pub statistics: Statistics,
    pub timestamp: String,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
    pub p99: f64,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub package_name: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub dependency_issues: Vec<String>,
    pub typescript_issues: Vec<String>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub package_name: String,
    pub suggestions: Vec<Suggestion>,
    pub potential_savings_kb: i64,
    pub estimated_performance_gain: f64,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub category: String,
    pub severity: String,
    pub description: String,
    pub impact: String,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub package_name: String,
    pub performance_delta: f64,
    pub memory_delta: f64,
    pub bundle_size_delta: f64,
    pub regression_detected: bool,
    pub improvements: Vec<String>,
    pub regressions: Vec<String>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_count: i32,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub os_version: String,
    pub architecture: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn discover_packages(workspace_path: &str) -> Result<Vec<String>> {
    use walkdir::WalkDir;

    let mut packages = Vec::new();

    for entry in WalkDir::new(workspace_path)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "package.json" {
            if let Some(parent) = entry.path().parent() {
                packages.push(parent.to_string_lossy().to_string());
            }
        }
    }

    Ok(packages)
}

fn calculate_delta(baseline: f64, current: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((current - baseline) / baseline) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_delta() {
        assert_eq!(calculate_delta(100.0, 110.0), 10.0);
        assert_eq!(calculate_delta(100.0, 90.0), -10.0);
        assert_eq!(calculate_delta(0.0, 100.0), 0.0);
    }

    #[test]
    fn test_system_info() {
        let info = get_system_info().unwrap();
        assert!(info.cpu_count > 0);
        assert!(info.total_memory_gb > 0.0);
    }
}
