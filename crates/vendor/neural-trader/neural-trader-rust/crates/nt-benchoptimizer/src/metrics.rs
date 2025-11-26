//! Metrics collection and analysis types

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub execution_time_ms: f64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub throughput_ops_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
}

impl BenchmarkMetrics {
    pub fn new() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_mb: 0,
            cpu_usage_percent: 0.0,
            throughput_ops_per_sec: 0.0,
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
        }
    }

    pub fn calculate_throughput(&mut self, operations: u64) {
        if self.execution_time_ms > 0.0 {
            self.throughput_ops_per_sec = (operations as f64 / self.execution_time_ms) * 1000.0;
        }
    }
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub package_name: String,
    pub passed: bool,
    pub failed_checks: Vec<String>,
    pub warnings: Vec<String>,
    pub dependency_graph_valid: bool,
    pub typescript_valid: bool,
    pub napi_bindings_valid: bool,
}

impl ValidationResult {
    pub fn new(package_name: String) -> Self {
        Self {
            package_name,
            passed: true,
            failed_checks: Vec::new(),
            warnings: Vec::new(),
            dependency_graph_valid: true,
            typescript_valid: true,
            napi_bindings_valid: true,
        }
    }

    pub fn add_failure(&mut self, check: String) {
        self.passed = false;
        self.failed_checks.push(check);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub priority: Priority,
    pub title: String,
    pub description: String,
    pub estimated_impact: Impact,
    pub implementation_effort: Effort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    BundleSize,
    Performance,
    Dependencies,
    CodeQuality,
    Security,
    Maintainability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impact {
    pub bundle_size_reduction_kb: Option<u64>,
    pub performance_improvement_percent: Option<f64>,
    pub memory_reduction_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effort {
    Trivial,    // < 1 hour
    Low,        // 1-4 hours
    Medium,     // 4-16 hours
    High,       // 16-40 hours
    VeryHigh,   // > 40 hours
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub hot_paths: Vec<HotPath>,
    pub memory_allocations: Vec<MemoryAllocation>,
    pub io_operations: Vec<IoOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPath {
    pub function_name: String,
    pub file_path: String,
    pub execution_time_ms: f64,
    pub call_count: u64,
    pub percentage_of_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub location: String,
    pub size_bytes: u64,
    pub allocation_count: u64,
    pub is_leaked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoOperation {
    pub operation_type: String,
    pub path: String,
    pub duration_ms: f64,
    pub bytes_transferred: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_metrics_default() {
        let metrics = BenchmarkMetrics::default();
        assert_eq!(metrics.execution_time_ms, 0.0);
        assert_eq!(metrics.memory_usage_mb, 0);
    }

    #[test]
    fn test_calculate_throughput() {
        let mut metrics = BenchmarkMetrics::default();
        metrics.execution_time_ms = 1000.0; // 1 second
        metrics.calculate_throughput(5000);
        assert_eq!(metrics.throughput_ops_per_sec, 5000.0);
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new("test-package".to_string());
        assert!(result.passed);

        result.add_failure("Missing dependency".to_string());
        assert!(!result.passed);
        assert_eq!(result.failed_checks.len(), 1);
    }
}
